"""Monokuler derinlik tahmini ve kutu-derinlik fuzyonu (Hafta 3).

Depth Anything V2, tek goruntuden **goreli ters derinlik** (disparity) uretir:
buyuk deger = kameraya yakin. Bu ayrim projenin en kolay karistirilan noktasi,
o yuzden kodda iki kavram ayri isimlerle tasiniyor:

    disparity   ham model ciktisi, buyuk = yakin
    distance    1 / disparity, buyuk = uzak (mesafeyle ayni yonde artar)

Mutlak metre olcusu tek kameradan cikarilamaz - bu monokuler gorunun bilinen
temel kisiti. Buradaki `distance` birimsizdir ve yalnizca nesneler arasi
karsilastirma icin anlamlidir. Olcek katsayisi Hafta 5'te sabit bir referansla
(serit genisligi ya da tipik arac genisligi) kalibre edilecek.
"""

from __future__ import annotations

import numpy as np

from .config import CameraConfig, DepthConfig
from .detection import Detection
from .utils import resolve_device, resolve_half

#: Bu degerin altindaki disparity gecersiz sayilir. Depth Anything'in ham
#: ciktisi cok uzak yuzeylerde sifirin altina inebiliyor (olculdu: -0.30);
#: 1/d oradan anlamli bir mesafe uretemez, bu yuzden "bilinmiyor" denir.
_MIN_DISPARITY = 1e-2


class DepthEstimator:
    """Depth Anything V2 sarmalayicisi.

    Model bir kez yuklenir ve isinma cagrisi kurucuda yapilir - Hafta 1'de
    ogrenildigi gibi, ilk cikarimin CUDA maliyeti olcum penceresine girmemeli.
    """

    def __init__(self, config: DepthConfig | None = None, camera: CameraConfig | None = None) -> None:
        self.config = config or DepthConfig()
        self.camera = camera or CameraConfig()
        self.device = resolve_device(self.config.device)

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - kurulum hatasi
            raise ImportError(
                "transformers kurulu degil. 'pip install -r requirements.txt' calistirin."
            ) from exc

        import torch

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.config.model)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.config.model)
        self._model.to(self.device)
        self._model.eval()

        self._use_half = resolve_half(self.config.half, self.device, layer="derinlik")
        if self._use_half:
            self._model.half()

        #: every_n_frames icin son uretilen harita
        self._cached: np.ndarray | None = None

        self.warmup()

    def warmup(self) -> None:
        """Modeli bir kez bos kareyle calistirir (bkz. YOLODetector.warmup)."""
        blank = np.zeros((self.config.input_width // 2, self.config.input_width, 3), dtype=np.uint8)
        self.infer(blank)

    def infer(self, image: np.ndarray, frame_index: int | None = None) -> np.ndarray:
        """BGR kareden goreli ters derinlik (disparity) haritasi uretir.

        `frame_index` verilir ve `config.every_n_frames > 1` ise, aradaki
        karelerde son harita yeniden kullanilir. Bu bir kestirme degil bilincli
        bir takas: derinlik sahnede yavas degisir, GPU yuku ise dogrudan
        bolunur.

        Donen harita girdi karesiyle ayni boyuttadir ve modelin **ham**
        ciktisidir; buyuk deger = yakin.

        Bilincli olarak normalize EDILMEZ. Kare bazinda [0,1]'e tasimak gorsel
        olarak cazip ama degerleri kareler arasi karsilastirilamaz kilar:
        sahneye tek bir yakin nesne girdiginde tum haritanin olcegi kayar ve
        hicbir sey hareket etmemis olsa bile her nesnenin "mesafesi" degisir.
        Hafta 5'teki hiz cikarimi tam da kareler arasi farka dayandigi icin bu
        kabul edilemez. Normalizasyon sadece cizim katmaninda, sadece renk
        haritasi icin yapilir.
        """
        import cv2

        stride = max(1, self.config.every_n_frames)
        if (
            frame_index is not None
            and stride > 1
            and self._cached is not None
            and frame_index % stride != 0
            and self._cached.shape == image.shape[:2]
        ):
            return self._cached

        source_h, source_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Boyut processor'e ACIKCA veriliyor. Verilmezse processor kendi
        # varsayilanina (518x518) gore yeniden olcekler ve onceden kucultulmus
        # kareyi bile geri buyutur - yani harici bir on-olcekleme modelin
        # maliyetini hic degistirmez. Olculdu: 518 -> 321 ms, 294 -> 109 ms.
        inputs = self._processor(
            images=rgb,
            return_tensors="pt",
            size=self._model_input_size(source_w, source_h),
        ).to(self.device)
        if self._use_half:
            inputs = inputs.to(self._torch.float16)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        small = outputs.predicted_depth.squeeze().float().cpu().numpy().astype(np.float32)
        # Buyutme cv2 ile: torch'un bicubic'iyle sonuc ayni (maks fark %0.001)
        # ama CPU'da 7 kat hizli (8.9 ms -> 1.3 ms).
        self._cached = cv2.resize(small, (source_w, source_h), interpolation=cv2.INTER_CUBIC)
        return self._cached

    def _model_input_size(self, source_w: int, source_h: int) -> dict[str, int]:
        """Modele verilecek tensor boyutu, en-boy orani korunarak.

        Depth Anything'in patch boyutu 14, bu yuzden her iki kenar da 14'un
        katina yuvarlanir.
        """
        width = max(14, int(round(self.config.input_width / 14)) * 14)
        height = max(14, int(round(width * source_h / source_w / 14)) * 14)
        return {"height": height, "width": width}


def normalize_for_display(disparity: np.ndarray, hood_top: float | None = None) -> np.ndarray:
    """Ham haritayi [0,1] araligina tasir - YALNIZCA cizim icin.

    Bu fonksiyonun ciktisi hicbir hesaba girmemeli; kare bazinda olcek
    degistirdigi icin kareler arasi karsilastirilamaz (bkz. DepthEstimator.infer).

    Kaput bolgesi istatistige katilmaz: kaput her karede sahnedeki en yakin
    yuzeydir ve maksimumu her zaman o belirler. Dahil edilirse yolun ve
    araclarin gercek araligi dar bir bantta sikisir, nesneler arasi fark
    renkten okunamaz hale gelir.
    """
    region = disparity
    if hood_top is not None:
        hood_row = int(round(disparity.shape[0] * hood_top))
        if hood_row > 0:
            region = disparity[:hood_row]

    low, high = float(region.min()), float(region.max())
    if high - low < 1e-6:
        return np.zeros_like(disparity)

    # Kaput bolgesi olcekten tasabilir (yol bolgesinden daha yakin), kirp.
    return np.clip((disparity - low) / (high - low), 0.0, 1.0)


def relative_distance(disparity: float) -> float | None:
    """Ters derinligi mesafeyle ayni yonde artan birimsiz bir sayiya cevirir.

    Gecersiz (sifir ya da negatif) disparity icin None doner. Tabana kirpip
    devasa bir sayi uretmek daha kolay olurdu ama o sayi olculmus gibi gorunur;
    None, "bu nesnenin derinligi bilinmiyor" demenin durust yolu.
    """
    if disparity < _MIN_DISPARITY:
        return None
    return 1.0 / disparity


def sample_region(
    detection: Detection,
    shape: tuple[int, int],
    config: DepthConfig,
    camera: CameraConfig | None = None,
) -> tuple[int, int, int, int] | None:
    """Bir tespitin derinliginin olculecegi piksel bolgesini dondurur.

    Kutunun **alt orta** bolgesi secilir: nesnenin zemine degdigi yer orasidir.
    Kutunun ust yarisi cogu zaman nesnenin arkasindaki sahneyi de kapsar ve
    oradan alinan derinlik nesneye degil arka plana aittir.

    Kaput sinirinin (camera.hood_top) altina tasan kisim kirpilir; o pikseller
    yolu degil aracin kendi kaputunu gosterir. Bolge tamamen kaputun altinda
    kalirsa None doner - o tespit icin derinlik uretilemez.

    Returns:
        (x1, y1, x2, y2) ya da gecerli bolge yoksa None.
    """
    height, width = shape
    bx1, by1, bx2, by2 = detection.xyxy

    box_w = bx2 - bx1
    box_h = by2 - by1
    if box_w <= 0 or box_h <= 0:
        return None

    cx = (bx1 + bx2) / 2
    half_w = box_w * config.sample_width_ratio / 2
    x1 = int(round(cx - half_w))
    x2 = int(round(cx + half_w))
    y1 = int(round(by2 - box_h * config.sample_height_ratio))
    y2 = int(round(by2))

    # Kaput sinirini uygula.
    if camera is not None and camera.hood_top is not None:
        hood_row = int(round(height * camera.hood_top))
        y2 = min(y2, hood_row)

    # Goruntu sinirlarina kirp.
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def fuse(
    detections: list[Detection],
    disparity_map: np.ndarray,
    config: DepthConfig,
    camera: CameraConfig | None = None,
) -> list[Detection]:
    """Her tespite derinlik degeri yazar (yerinde degistirir).

    Ornek bolgesindeki degerlerin **medyani** alinir, ortalamasi degil:
    kutu icinde arka plana ait birkac piksel her zaman bulunur ve ortalama
    bunlardan etkilenir, medyan etkilenmez.
    """
    shape = disparity_map.shape[:2]

    for det in detections:
        region = sample_region(det, shape, config, camera)
        if region is None:
            det.depth = None
            continue

        x1, y1, x2, y2 = region
        patch = disparity_map[y1:y2, x1:x2]
        if patch.size == 0:
            det.depth = None
            continue

        det.depth = relative_distance(float(np.median(patch)))

    return detections
