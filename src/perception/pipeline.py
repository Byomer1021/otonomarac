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

from .bev import BEVProjector
from .config import Config
from .depth import DepthEstimator, fuse, normalize_for_display
from .detection import Detection, YOLODetector
from .risk import RiskEstimator
from .tracking import ObjectTracker
from .utils import Profiler, describe_device
from .video_io import Frame, VideoReader, VideoWriter
from .visualize import colorize_depth, draw_detections, draw_hud, draw_trails, stack_panels


@dataclass
class FrameResult:
    """Tek bir karenin isleme sonucu."""

    frame: Frame
    #: Tespit katmaninin ham ciktisi
    detections: list[Detection] = field(default_factory=list)
    #: Kimlik atanmis nesneler. Takip kapaliysa `detections` ile aynidir.
    tracked: list[Detection] = field(default_factory=list)
    #: iz kimligi -> gecmis (kare_no, x, y) zemin temas noktalari
    trails: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)
    #: Goreli ters derinlik haritasi ([0,1], buyuk = yakin). Katman kapaliysa None.
    disparity: np.ndarray | None = None
    #: Cizim yapilmis kare (gorsellestirme kapaliysa None)
    rendered: np.ndarray | None = None

    @property
    def objects(self) -> list[Detection]:
        """Sonraki katmanlarin uzerinde calisacagi nesne listesi."""
        return self.tracked or self.detections


class PerceptionPipeline:
    """Video karelerini algi katmanlarindan gecirir."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

        for warning in self.config.validate():
            print(f"UYARI: {warning}\n")

        self.profiler = Profiler()
        self.detector = YOLODetector(self.config.detection)
        self.tracker = ObjectTracker(self.config.tracking) if self.config.tracking.enabled else None
        self.depth = (
            DepthEstimator(self.config.depth, self.config.camera) if self.config.depth.enabled else None
        )
        self.bev = (
            BEVProjector(self.config.bev, self.config.camera)
            if self.config.bev.enabled and self.config.camera.road_quad
            else None
        )
        # Risk, zemin konumuna dayaniyor: BEV yoksa hesaplanamaz.
        self.risk = (
            RiskEstimator(self.config.risk)
            if self.config.risk.enabled and self.bev is not None
            else None
        )

    @property
    def device_label(self) -> str:
        return describe_device(self.detector.device)

    def process_frame(self, frame: Frame) -> FrameResult:
        """Tek kareyi isler. Yeni katmanlar bu metoda sirayla eklenecek."""
        with self.profiler.stage("tespit"):
            detections = self.detector.detect(frame.image)

        result = FrameResult(frame=frame, detections=detections)

        if self.tracker is not None:
            with self.profiler.stage("takip"):
                result.tracked = self.tracker.update(detections, frame.index)
                result.trails = self.tracker.active_trails(result.tracked)

        if self.depth is not None:
            with self.profiler.stage("derinlik"):
                result.disparity = self.depth.infer(frame.image, frame.index)
            # Fuzyon ayri olculuyor: derinlik cikarimi GPU'da, fuzyon CPU'da
            # calisiyor ve ikisini tek sayida toplamak hangisinin darbogaz
            # oldugunu gizlerdi.
            with self.profiler.stage("fuzyon"):
                fuse(result.objects, result.disparity, self.config.depth, self.config.camera)

        if self.bev is not None:
            with self.profiler.stage("projeksiyon"):
                self.bev.project(result.objects, frame.image.shape[:2])

        if self.risk is not None:
            with self.profiler.stage("risk"):
                # Kare numarasi degil zaman damgasi: izlerin bir kismi
                # okluzyonda kaybolup geri geliyor ve gozlemler arasi kare
                # sayisi sabit degil.
                self.risk.update(result.objects, frame.timestamp, self.bev.ego_offset_m)

        return result

    def render(self, result: FrameResult, fps: float | None = None) -> np.ndarray:
        """Sonucu kare uzerine cizer ve cizilmis kareyi dondurur."""
        with self.profiler.stage("cizim"):
            canvas = result.frame.image.copy()
            objects = result.objects

            # Izler kutulardan once cizilir; aksi halde kutu kenarlarinin
            # uzerinden gecip goruntuyu kirletir.
            if self.config.visualize.show_trails and result.trails:
                classes = {d.track_id: d.cls_id for d in objects if d.track_id is not None}
                draw_trails(canvas, result.trails, classes)

            draw_detections(canvas, objects, self.config.visualize)

            if self.config.visualize.show_hud:
                lines = [f"Kare {result.frame.index}  |  {result.frame.timestamp:5.2f}s"]
                if self.tracker is None:
                    lines.append(f"Nesne: {len(objects)}")
                else:
                    # Iki sayinin birlikte artmasi saglikli; toplam kimlik
                    # aktif iz sabitken hizla buyuyorsa kimlik atlamasi var.
                    lines.append(
                        f"Aktif iz: {len(objects)}  |  Toplam kimlik: {len(self.tracker.stats.frames_seen)}"
                    )
                lines.append(f"Cihaz: {self.device_label}")
                if fps is not None:
                    lines.insert(0, f"{fps:.1f} FPS")
                draw_hud(canvas, lines)

            if self.config.depth.show_panel and result.disparity is not None:
                canvas = stack_panels(
                    canvas,
                    # Normalizasyon SADECE burada: renk haritasi icin. Hesaba
                    # giren degerler ham disparity uzerinden uretiliyor.
                    colorize_depth(
                        normalize_for_display(result.disparity, self.config.camera.hood_top),
                        self.config.depth.colormap,
                    ),
                    # Yon acikca yaziliyor: harita ters derinlik, yani acik
                    # renk yakin demek. Tersini varsaymak kolay bir hata.
                    labels=("Kamera", "Derinlik (acik = yakin, goreli)"),
                )

            if self.bev is not None:
                canvas = stack_panels(canvas, self.bev.render(result.objects))

        result.rendered = canvas
        return canvas

    def analyze(self, input_path: str | Path | None = None) -> None:
        """Videoyu isler ama cizim ve yazma yapmaz.

        Parametre taramasi icin: tek amac takip istatistiklerini toplamak
        oldugunda kare kare mp4 yazmak zamanin buyuk kismini yiyor.
        """
        source = input_path or self.config.video.input
        if source is None:
            raise ValueError("Girdi videosu belirtilmedi.")

        with VideoReader(
            source,
            frame_stride=self.config.video.frame_stride,
            max_frames=self.config.video.max_frames,
            resize_width=self.config.video.resize_width,
        ) as reader:
            if self.tracker is not None:
                self.tracker.configure_for_fps(reader.output_meta.fps)
            for frame in reader:
                with self.profiler.stage("kare_toplam"):
                    self.process_frame(frame)

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
            if self.tracker is not None:
                self.tracker.configure_for_fps(meta.fps)
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

        if self.tracker is not None:
            print("\nTakip analizi:")
            print(self.tracker.stats.format_report(self.config.tracking.min_track_len))

        if self.risk is not None:
            print("\nRisk analizi:")
            print(self.risk.summary())

        print("\nPerformans:")
        print(self.profiler.format_table())
        return target

    def _live_fps(self) -> float | None:
        """HUD'da gostermek icin o ana kadarki ortalama isleme hizi."""
        mean_ms = self.profiler.mean_ms("kare_toplam")
        return 1000.0 / mean_ms if mean_ms > 0 else None
