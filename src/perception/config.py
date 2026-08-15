"""Pipeline yapilandirmasi.

YAML dosyasindan okunur, CLI argumanlariyla ezilebilir. Her hafta yeni bir
katman eklendikce buraya yeni bir alt-config dataclass'i eklenecek
(TrackingConfig, DepthConfig, BEVConfig, RiskConfig ...) ve `Config` icine
alan olarak yazilacak - baska bir yeri degistirmek gerekmez.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoConfig:
    """Video giris/cikis ve on-isleme ayarlari."""

    input: str | None = None
    output: str = "outputs/result.mp4"
    #: Kac karede bir isleme alinacak (1 = her kare). CPU'da hizlandirmak icin.
    frame_stride: int = 1
    #: Sadece ilk N kareyi isle (hizli deneme icin). None = tamami.
    max_frames: int | None = None
    #: Isleme oncesi kareyi bu genislige olcekle (en-boy orani korunur). None = dokunma.
    resize_width: int | None = 1280


@dataclass
class DetectionConfig:
    """YOLO tespit katmani ayarlari."""

    model: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    #: "auto" | "cpu" | "cuda" | "cuda:0"
    device: str = "auto"
    #: COCO sinif id'leri: person, bicycle, car, motorcycle, bus, truck, traffic light
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 9])
    #: Yari hassasiyet (GPU'da hizlandirir, CPU'da yok sayilir)
    half: bool = False


@dataclass
class VisualizeConfig:
    """Cizim ayarlari."""

    show_conf: bool = True
    show_class_name: bool = True
    show_hud: bool = True
    line_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class Config:
    """Tum pipeline yapilandirmasi."""

    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """YAML dosyasindan config uretir. `path` None ise varsayilanlar kullanilir."""
        if path is None:
            return cls()

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config dosyasi bulunamadi: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config dosyasinin koku bir sozluk olmali: {path}")

        cfg = cls()
        for section_name, values in raw.items():
            section = getattr(cfg, section_name, None)
            if not is_dataclass(section):
                warnings.warn(f"Config'de bilinmeyen bolum yok sayildi: {section_name}", stacklevel=2)
                continue
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"'{section_name}' bolumu bir sozluk olmali, {type(values).__name__} geldi")
            _apply(section, values, section_name)
        return cfg

    def override(self, **kwargs: Any) -> Config:
        """`bolum.alan=deger` bicimindeki duz anahtarlarla config'i gunceller.

        None degerler yok sayilir; boylece CLI'da "arguman verilmedi" durumu
        ayrica kontrol edilmek zorunda kalmaz.
        """
        for dotted, value in kwargs.items():
            if value is None:
                continue
            section_name, _, key = dotted.partition(".")
            section = getattr(self, section_name, None)
            if not key or not is_dataclass(section):
                raise KeyError(f"Gecersiz override anahtari: {dotted}")
            if not hasattr(section, key):
                raise KeyError(f"Bilinmeyen config alani: {dotted}")
            setattr(section, key, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        """Calistirilan config'i diske yazar (tekrar uretilebilirlik icin)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _apply(section: Any, values: dict[str, Any], section_name: str) -> None:
    """Bir alt-config dataclass'inin alanlarini sozlukten gunceller."""
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            warnings.warn(f"Config'de bilinmeyen anahtar yok sayildi: {section_name}.{key}", stacklevel=3)
            continue
        setattr(section, key, value)
