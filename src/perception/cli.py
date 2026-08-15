"""Komut satiri arayuzu.

Ornekler:
    python -m perception.cli --input data/test.mp4
    python -m perception.cli --input data/test.mp4 --max-frames 100 --model yolov8s.pt
    python -m perception.cli --config configs/default.yaml --input data/test.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .pipeline import PerceptionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perception",
        description="Tek kameradan surus sahnesi algilama pipeline'i",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config dosyasi")
    parser.add_argument("-i", "--input", type=str, default=None, help="Girdi videosu")
    parser.add_argument("-o", "--output", type=str, default=None, help="Cikti videosu")

    video = parser.add_argument_group("video")
    video.add_argument("--frame-stride", type=int, default=None, help="Kac karede bir islensin")
    video.add_argument("--max-frames", type=int, default=None, help="Sadece ilk N kareyi isle")
    video.add_argument("--resize-width", type=int, default=None, help="Isleme genisligi (piksel)")

    detection = parser.add_argument_group("tespit")
    detection.add_argument("--model", type=str, default=None, help="YOLO agirlik dosyasi")
    detection.add_argument("--conf", type=float, default=None, help="Guven esigi")
    detection.add_argument("--imgsz", type=int, default=None, help="Model giris boyutu")
    detection.add_argument(
        "--device", type=str, default=None, help="auto | cpu | cuda | cuda:0"
    )
    detection.add_argument("--half", action="store_true", default=None, help="FP16 (sadece GPU)")

    tracking = parser.add_argument_group("takip")
    tracking.add_argument(
        "--no-track",
        action="store_true",
        help="Takibi kapat, sadece tespit. Varsayilan conf takip icin dusuk "
        "tutuldugundan bunu --conf 0.35 ile birlikte kullanin",
    )
    tracking.add_argument("--track-buffer", type=int, default=None, help="Kayip izin yasatilacagi kare")
    tracking.add_argument("--match-thresh", type=float, default=None, help="IoU eslestirme esigi")
    tracking.add_argument(
        "--trail-seconds", type=float, default=None, help="Hareket izinin gecmisi, saniye (0 = kapat)"
    )

    parser.add_argument("--no-hud", action="store_true", help="Bilgi panelini gizle")
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Calistirilan config'i cikti klasorune yaz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = Config.load(args.config).override(
        **{
            "video.input": args.input,
            "video.output": args.output,
            "video.frame_stride": args.frame_stride,
            "video.max_frames": args.max_frames,
            "video.resize_width": args.resize_width,
            "detection.model": args.model,
            "detection.conf": args.conf,
            "detection.imgsz": args.imgsz,
            "detection.device": args.device,
            "detection.half": args.half,
            "tracking.enabled": False if args.no_track else None,
            "tracking.track_buffer": args.track_buffer,
            "tracking.match_thresh": args.match_thresh,
            "tracking.trail_seconds": args.trail_seconds,
            "visualize.show_hud": False if args.no_hud else None,
        }
    )

    try:
        pipeline = PerceptionPipeline(config)
        output = pipeline.run()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    if args.dump_config:
        config_path = output.with_suffix(".config.yaml")
        config.dump(config_path)
        print(f"Config yazildi -> {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
