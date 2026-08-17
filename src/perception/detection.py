"""Nesne tespit katmani (Hafta 1).

Ultralytics YOLO sarmalayicisi. Pipeline'in geri kalani Ultralytics'in
`Results` nesnesini hic gormez - sadece asagidaki `Detection` dataclass'ini
gorur. Bu sayede ileride model degistirmek (YOLO -> RT-DETR vb.) tek
dosyalik bir is olur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import DetectionConfig
from .utils import resolve_device, resolve_half

if TYPE_CHECKING:
    from ultralytics import YOLO


#: Bu projede ilgilendigimiz COCO siniflari (id -> okunabilir ad).
CLASS_NAMES: dict[int, str] = {
    0: "yaya",
    1: "bisiklet",
    2: "araba",
    3: "motosiklet",
    5: "otobus",
    7: "kamyon",
    9: "trafik isigi",
}

#: Kusbakisi haritada "arac" olarak islenecek siniflar (Hafta 4'te kullanilacak).
VEHICLE_CLASSES: frozenset[int] = frozenset({2, 3, 5, 7})

#: Kirilgan yol kullanicilari - risk katmaninda ayri esik uygulanacak (Hafta 5).
VULNERABLE_CLASSES: frozenset[int] = frozenset({0, 1})


@dataclass
class Detection:
    """Tek bir kareye ait tek bir tespit.

    Sonraki haftalarda buyuyecek alanlar simdiden None olarak duruyor:
    `track_id` Hafta 2'de, `depth` Hafta 3'te, `bev_xy` Hafta 4'te dolar.
    """

    #: (x1, y1, x2, y2) piksel koordinatlari
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    track_id: int | None = None
    depth: float | None = None
    bev_xy: tuple[float, float] | None = None

    @property
    def cls_name(self) -> str:
        return CLASS_NAMES.get(self.cls_id, f"sinif_{self.cls_id}")

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Kutunun alt orta noktasi - nesnenin zemine degdigi varsayilan nokta.

        Kusbakisi projeksiyonun (Hafta 4) girdisi tam olarak budur.
        """
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vehicle(self) -> bool:
        return self.cls_id in VEHICLE_CLASSES


class YOLODetector:
    """Ultralytics YOLO tabanli tespit edici.

    Model, ilk cagrida degil kurucu icinde yuklenir; boylece agirlik indirme
    gibi yavas isler video isleme dongusunun icine sizmaz.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.device = resolve_device(self.config.device)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - kurulum hatasi
            raise ImportError(
                "ultralytics kurulu degil. 'pip install -r requirements.txt' calistirin."
            ) from exc

        self.model: YOLO = YOLO(self.config.model)
        self.model.to(self.device)

        # Ultralytics 8.4 ile `half` bayragi `quantize` altinda birlestirildi
        # (16 = FP16, None = FP32); eski adi gecirmek deprecation uyarisi
        # bastiriyor. resolve_half ayrica donanimi denetliyor - Pascal'da FP16
        # daha yavas oldugu icin istek reddediliyor.
        self._quantize = 16 if resolve_half(self.config.half, self.device, layer="tespit") else None

        self.warmup()

    def warmup(self) -> None:
        """Modeli bir kez bos kareyle calistirir.

        Ilk cikarim CUDA kernel derlemesi ve bellek ayirma yuzunden saniyeler
        surer. Bu maliyet olcum dongusunun icinde kalirsa performans tablosu
        gercekte olmayan bir yavasligi raporlar - bu yuzden dongu baslamadan
        once burada odenir.
        """
        blank = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        self.detect(blank)

    def __call__(self, image: np.ndarray) -> list[Detection]:
        return self.detect(image)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Tek kare uzerinde tespit calistirir."""
        results = self.model.predict(
            image,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            classes=self.config.classes or None,
            device=self.device,
            quantize=self._quantize,
            verbose=False,
        )
        return self._to_detections(results[0])

    @staticmethod
    def _to_detections(result) -> list[Detection]:
        """Ultralytics `Results` -> `Detection` listesi."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        # Takip acikken (Hafta 2) Ultralytics kimlikleri de doldurur.
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        return [
            Detection(
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(c),
                cls_id=int(k),
                track_id=int(ids[i]) if ids is not None else None,
            )
            for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(xyxy, conf, cls))
        ]
