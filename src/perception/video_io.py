"""Video okuma / yazma katmani.

Pipeline'in geri kalani OpenCV'nin VideoCapture ayrintilarini gormesin diye
ince bir sarmalayici. Kare atlama ve olcekleme burada yapilir; boylece
"CPU'da yavas kaliyor" riskine karsi tek bir mudahale noktasi olur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    """Kaynak videonun temel ozellikleri."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    """Islenecek tek kare ve kaynaktaki konumu."""

    #: Kaynak videodaki gercek kare indeksi (atlananlar dahil sayilir)
    index: int
    #: Kaynak videodaki zaman damgasi (saniye)
    timestamp: float
    image: np.ndarray


class VideoReader:
    """Videoyu kare kare okur; istege bagli kare atlama ve olcekleme uygular.

    Context manager olarak kullanilir:

        with VideoReader("in.mp4", resize_width=1280) as reader:
            for frame in reader:
                ...
    """

    def __init__(
        self,
        path: str | Path,
        *,
        frame_stride: int = 1,
        max_frames: int | None = None,
        resize_width: int | None = None,
    ) -> None:
        if frame_stride < 1:
            raise ValueError(f"frame_stride >= 1 olmali, {frame_stride} verildi")

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video bulunamadi: {self.path}")

        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.resize_width = resize_width

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Video acilamadi: {self.path}. Kodek desteklenmiyor olabilir "
                f"(H.264 mp4 ile deneyin)."
            )

        src_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.source = VideoMeta(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Bazi dosyalarda FPS 0 gelir; makul bir varsayilana duselim.
            fps=src_fps if src_fps and src_fps > 0 else 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @property
    def output_meta(self) -> VideoMeta:
        """Islenmis karelerin ozellikleri - cikti yazicisi bunu kullanmali."""
        width, height = self.source.width, self.source.height
        if self.resize_width and width > 0:
            scale = self.resize_width / width
            width, height = self.resize_width, max(1, int(round(height * scale)))

        expected = self.source.frame_count // self.frame_stride if self.source.frame_count > 0 else 0
        if self.max_frames is not None:
            expected = min(expected, self.max_frames) if expected else self.max_frames

        return VideoMeta(
            width=width,
            height=height,
            # Kare atlanirsa cikti videosu gercek zamanli kalsin diye FPS de bolunur.
            fps=self.source.fps / self.frame_stride,
            frame_count=expected,
        )

    def __iter__(self) -> Iterator[Frame]:
        source_index = 0
        yielded = 0

        while True:
            ok, image = self._cap.read()
            if not ok:
                break

            if source_index % self.frame_stride == 0:
                if self.resize_width and image.shape[1] != self.resize_width:
                    scale = self.resize_width / image.shape[1]
                    new_size = (self.resize_width, max(1, int(round(image.shape[0] * scale))))
                    # INTER_AREA kucultmede en temiz sonucu verir.
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

                yield Frame(
                    index=source_index,
                    timestamp=source_index / self.source.fps,
                    image=image,
                )
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break

            source_index += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class VideoWriter:
    """Karesi geldikce cikti videosuna yazar.

    Boyut ilk karede kilitlenir; sonraki karelerde farkli boyut gelirse hata verir
    (sessizce bozuk video uretmektense patlamasi iyidir).
    """

    def __init__(self, path: str | Path, *, fps: float, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps > 0 else 30.0
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def write(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]

        if self._writer is None:
            self._size = (width, height)
            self._writer = cv2.VideoWriter(str(self.path), self._fourcc, self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(f"Cikti videosu acilamadi: {self.path}")
        elif self._size != (width, height):
            raise ValueError(
                f"Kare boyutu degisti: beklenen {self._size}, gelen {(width, height)}"
            )

        self._writer.write(image)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
