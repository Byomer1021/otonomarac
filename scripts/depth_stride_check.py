"""Derinlik yeniden kullaniminin nesne basina hataya etkisini olcer (Hafta 3).

`depth.every_n_frames` GPU yukunu dogrudan boluyor, ama bedeli olculmeden
varsayilan yapilamaz. Betik ayni kareler uzerinde k=1 (her kare) sonucunu
referans alip k>1 sonuclariyla karsilastirir.

Karsilastirma nesne bazinda ve goreli yapilir: her (kare, iz kimligi) cifti
icin |d_k - d_1| / d_1. Mutlak fark anlamsiz olurdu cunku derinlik birimsiz
ve uzak nesnelerde deger buyur.

Kullanim:
    python scripts/depth_stride_check.py data/maltepe_test.mp4 --frames 90
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception.config import Config  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402


def collect(video: Path, base: Config, stride: int, frames: int) -> dict[tuple[int, int], float]:
    """(kare, iz kimligi) -> derinlik esleme uretir."""
    config = Config()
    config.video = replace(base.video, input=str(video), max_frames=frames)
    config.camera = replace(base.camera)
    config.detection = replace(base.detection)
    config.tracking = replace(base.tracking)
    config.depth = replace(base.depth, every_n_frames=stride, show_panel=False)
    config.visualize = replace(base.visualize)

    pipeline = PerceptionPipeline(config)
    depths: dict[tuple[int, int], float] = {}

    from perception.video_io import VideoReader

    with VideoReader(
        video,
        frame_stride=config.video.frame_stride,
        max_frames=frames,
        resize_width=config.video.resize_width,
    ) as reader:
        if pipeline.tracker is not None:
            pipeline.tracker.configure_for_fps(reader.output_meta.fps)
        for frame in reader:
            result = pipeline.process_frame(frame)
            for det in result.objects:
                if det.track_id is not None and det.depth is not None:
                    depths[(frame.index, det.track_id)] = det.depth

    return depths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=90, help="Islenecek kare sayisi")
    parser.add_argument("--strides", type=str, default="2,3,5", help="Denenecek k degerleri")
    args = parser.parse_args(argv)

    if not args.video.is_file():
        raise SystemExit(f"Video bulunamadi: {args.video}")

    base = Config.load(args.config)
    strides = [int(s) for s in args.strides.split(",")]

    print(f"\n{args.video.name}, ilk {args.frames} kare\n")
    print("k=1 (referans) hesaplaniyor...", flush=True)
    reference = collect(args.video, base, 1, args.frames)
    print(f"  {len(reference)} nesne-kare olcumu\n")

    header = f"{'k':>3} | {'ortak olcum':>11} | {'medyan hata':>11} | {'90. yuzdelik':>12} | {'maks':>7}"
    print(header)
    print("-" * len(header))
    print(f"{1:>3} | {len(reference):>11} | {'0.0%':>11} | {'0.0%':>12} | {'0.0%':>7}")

    for k in strides:
        current = collect(args.video, base, k, args.frames)
        shared = sorted(set(reference) & set(current))
        if not shared:
            print(f"{k:>3} | {'ortak olcum yok':>11}")
            continue

        errors = sorted(abs(current[key] - reference[key]) / reference[key] * 100 for key in shared)
        p90 = errors[int(len(errors) * 0.9)] if len(errors) > 1 else errors[0]
        print(
            f"{k:>3} | {len(shared):>11} | {statistics.median(errors):>10.1f}% | "
            f"{p90:>11.1f}% | {max(errors):>6.1f}%"
        )

    print(
        "\nHata, nesnenin kendi derinligine oranla. Medyan birkac yuzde ise\n"
        "yeniden kullanim guvenli; kuyruk (maks) hizli yaklasan nesnelerde buyur."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
