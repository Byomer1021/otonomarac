"""Yol segmentasyonu ve serit boyasi cikarimi (Hafta 6).

Cityscapes uzerinde egitilmis SegFormer, piksel basina 19 sinif uretir;
bizim ihtiyacimiz olan `road` (id 0) ve `sidewalk` (id 1).

Iki tasarim notu:

**Kaput kesiliyor.** Kaput kadrajin alt %15'ini kapliyor ve model onu yol
sanıyor - olculdu, kaput dahilken "yol" pikselinin buyuk kismi araca aitti.
Kesildiginde kadraj Cityscapes'in egitildigi cerceveye de yaklasiyor;
kaputun ustundeki yol orani %3.5'ten %6.4'e cikti.

**Serit boyasi maskeyle kapilaniyor.** Hafta 4'te serit cizgilerini renkten
bulmaya calisildi ve basarisiz oldu: beyaz maske kaldirimi, park etmis
araclari ve binalari da yakaliyordu. Artik ayni renk esigi yalnizca yol
maskesinin icinde uygulaniyor - o hafta bırakılan is burada kapaniyor.
"""

from __future__ import annotations

import numpy as np

from .config import CameraConfig, SegmentationConfig
from .utils import resolve_device, resolve_half

#: Cityscapes sinif kimlikleri.
ROAD_ID = 0
SIDEWALK_ID = 1

#: Serit boyasi HSV araliklari. Sari orta cizgi ve beyaz kenar/serit cizgisi.
_YELLOW = ((15, 80, 110), (38, 255, 255))
_WHITE = ((0, 0, 170), (180, 50, 255))


class RoadSegmenter:
    """SegFormer sarmalayicisi: yol maskesi + serit boyasi."""

    def __init__(
        self, config: SegmentationConfig | None = None, camera: CameraConfig | None = None
    ) -> None:
        self.config = config or SegmentationConfig()
        self.camera = camera or CameraConfig()
        self.device = resolve_device(self.config.device)

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers kurulu degil. 'pip install -r requirements.txt' calistirin."
            ) from exc

        import torch

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.config.model)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.config.model)
        self._model.to(self.device)
        self._model.eval()

        self._use_half = resolve_half(self.config.half, self.device, layer="segmentasyon")
        if self._use_half:
            self._model.half()

        self._cached: np.ndarray | None = None
        self.warmup()

    def warmup(self) -> None:
        """Modeli bir kez bos kareyle calistirir (bkz. YOLODetector.warmup)."""
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        self.segment(blank)

    # ---------- segmentasyon ----------

    def segment(self, image: np.ndarray, frame_index: int | None = None) -> np.ndarray:
        """BGR kareden yol maskesi uretir (uint8, yol=1).

        Maske girdi karesiyle ayni boyuttadir. Kaputun altindaki satirlar
        her zaman 0'dir: orasi yol degil aracin kendisi.
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

        height, width = image.shape[:2]
        hood_row = (
            int(round(height * self.camera.hood_top)) if self.camera.hood_top else height
        )
        crop = image[:hood_row]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        w = max(32, int(round(self.config.input_width / 32)) * 32)
        h = max(32, int(round(w * crop.shape[0] / crop.shape[1] / 32)) * 32)
        inputs = self._processor(
            images=rgb, return_tensors="pt", size={"height": h, "width": w}
        ).to(self.device)
        if self._use_half:
            inputs = inputs.to(self._torch.float16)

        with self._torch.inference_mode():
            logits = self._model(**inputs).logits

        small = logits.argmax(1)[0].to(self._torch.uint8).cpu().numpy()
        labels = cv2.resize(small, (width, hood_row), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:hood_row] = (labels == ROAD_ID).astype(np.uint8)

        if self.config.close_kernel > 1:
            k = np.ones((self.config.close_kernel, self.config.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        self._cached = mask
        return mask

    # ---------- serit boyasi ----------

    def lane_paint(self, image: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """Yol maskesi icinde kalan serit boyasini bulur (uint8, boya=1).

        Hafta 4'te bu, maske olmadan denenmis ve calismamisti: parlak her
        yuzey - kaldirim, beyaz araclar, bina cepheleri - beyaz esigine
        takiliyordu. Yol maskesiyle kapilaninca ayni esik ise yariyor.
        """
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        paint = cv2.bitwise_or(
            cv2.inRange(hsv, *_YELLOW), cv2.inRange(hsv, *_WHITE)
        )
        paint = cv2.bitwise_and(paint, paint, mask=road_mask)
        # Tekil pikselleri at: boya cizgi olusturur, benek degil.
        k = np.ones((3, 3), np.uint8)
        paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, k)
        return (paint > 0).astype(np.uint8)
