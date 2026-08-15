"""Algi pipeline'i - katmanlarin bir araya getirildigi yer.

Hafta 1'de sadece tespit katmani var. Sonraki haftalarda `process_frame`
icine sirayla takip, derinlik, fuzyon, projeksiyon ve risk katmanlari
eklenecek; `FrameResult` de o katmanlarin ciktilariyla buyuyecek.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from .detection import Detection, YOLODetector
from .utils import Profiler, describe_device
from .video_io import Frame, VideoReader, VideoWriter
from .visualize import draw_detections, draw_hud


@dataclass
class FrameResult:
    """Tek bir karenin isleme sonucu."""

    frame: Frame
    detections: list[Detection] = field(default_factory=list)
    #: Cizim yapilmis kare (gorsellestirme kapaliysa None)
    rendered: np.ndarray | None = None


class PerceptionPipeline:
    """Video karelerini algi katmanlarindan gecirir."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.profiler = Profiler()
        self.detector = YOLODetector(self.config.detection)

    @property
    def device_label(self) -> str:
        return describe_device(self.detector.device)

    def process_frame(self, frame: Frame) -> FrameResult:
        """Tek kareyi isler. Yeni katmanlar bu metoda sirayla eklenecek."""
        with self.profiler.stage("tespit"):
            detections = self.detector.detect(frame.image)

        return FrameResult(frame=frame, detections=detections)

    def render(self, result: FrameResult, fps: float | None = None) -> np.ndarray:
        """Sonucu kare uzerine cizer ve cizilmis kareyi dondurur."""
        with self.profiler.stage("cizim"):
            canvas = result.frame.image.copy()
            draw_detections(canvas, result.detections, self.config.visualize)

            if self.config.visualize.show_hud:
                lines = [
                    f"Kare {result.frame.index}  |  {result.frame.timestamp:5.2f}s",
                    f"Nesne: {len(result.detections)}",
                    f"Cihaz: {self.device_label}",
                ]
                if fps is not None:
                    lines.insert(0, f"{fps:.1f} FPS")
                draw_hud(canvas, lines)

        result.rendered = canvas
        return canvas

    def run(self, input_path: str | Path | None = None, output_path: str | Path | None = None) -> Path:
        """Videoyu bastan sona isler ve cikti videosunun yolunu dondurur."""
        video_cfg = self.config.video
        source = input_path or video_cfg.input
        if source is None:
            raise ValueError("Girdi videosu belirtilmedi (--input veya config.video.input).")
        target = Path(output_path or video_cfg.output)

        with VideoReader(
            source,
            frame_stride=video_cfg.frame_stride,
            max_frames=video_cfg.max_frames,
            resize_width=video_cfg.resize_width,
        ) as reader:
            meta = reader.output_meta
            print(f"Girdi : {reader.path}")
            print(
                f"        {reader.source.width}x{reader.source.height} @ "
                f"{reader.source.fps:.1f} FPS, {reader.source.frame_count} kare"
            )
            print(f"Cikti : {target}  ({meta.width}x{meta.height} @ {meta.fps:.1f} FPS)")
            print(f"Model : {self.config.detection.model} -> {self.device_label}\n")

            with VideoWriter(target, fps=meta.fps) as writer:
                progress = tqdm(
                    reader,
                    total=meta.frame_count or None,
                    unit="kare",
                    desc="Isleniyor",
                )
                for frame in progress:
                    with self.profiler.stage("kare_toplam"):
                        result = self.process_frame(frame)
                        canvas = self.render(result, fps=self._live_fps())
                        writer.write(canvas)

                frames_written = writer.frames_written

        print(f"\n{frames_written} kare yazildi -> {target.resolve()}")
        print("\nPerformans:")
        print(self.profiler.format_table())
        return target

    def _live_fps(self) -> float | None:
        """HUD'da gostermek icin o ana kadarki ortalama isleme hizi."""
        mean_ms = self.profiler.mean_ms("kare_toplam")
        return 1000.0 / mean_ms if mean_ms > 0 else None
