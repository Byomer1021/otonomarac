"""Takip parametrelerini tarar ve kimlik surekliligini karsilastirir (Hafta 2).

Ilk calistirmada takip "calisiyor" gorunuyordu ama olcum baska sey soyledi:
154 karede 5.4 es zamanli nesne icin 55 benzersiz kimlik uretilmisti ve
izlerin yarisi 5 kareden kisaydi. Hangi parametrenin ne kadar etkisi oldugunu
tahmin etmek yerine olcmek icin bu betik yazildi.

Kullanim:
    python scripts/tracking_sweep.py data/kitti_0005.mp4
    python scripts/tracking_sweep.py data/kitti_0005.mp4 --only baz,model_s
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

# Depoyu editable kurmadan da calissin.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception.config import Config  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402

#: Her deney: ad -> (aciklama, detection ezmeleri, tracking ezmeleri)
#: Baz disindaki her satir bazdan TEK bir eksende ayrilir; boylece farkin
#: hangi degisiklikten geldigi belirsiz kalmaz.
EXPERIMENTS: dict[str, tuple[str, dict, dict]] = {
    "baz": ("varsayilan ayarlar", {}, {}),
    "model_s": ("yolov8s (daha buyuk model)", {"model": "yolov8s.pt"}, {}),
    "imgsz_960": ("giris 960px (kucuk nesneler)", {"imgsz": 960}, {}),
    "conf_015": ("detektor esigi 0.15", {"conf": 0.15}, {}),
    "buffer_60": ("kayip iz 60 kare yasasin", {}, {"track_buffer": 60}),
    "match_090": ("eslestirme esigi 0.9", {}, {"match_thresh": 0.9}),
    "newtrack_040": ("yeni iz esigi 0.40", {}, {"new_track_thresh": 0.40}),
    # Tek eksenli deneylerin en iyi ikisinin birlesimi.
    "birlesik": ("yolov8s + eslestirme 0.9", {"model": "yolov8s.pt"}, {"match_thresh": 0.9}),
}


def run_one(video: Path, base: Config, det_over: dict, trk_over: dict) -> dict[str, float]:
    config = Config()
    config.video = replace(base.video, input=str(video))
    config.detection = replace(base.detection, **det_over)
    config.tracking = replace(base.tracking, **trk_over)

    # Zamanlayici pipeline kurulduktan SONRA baslar. Kurulum agirlik indirme
    # ve CUDA warmup iceriyor; bunlar olcume karisirsa ilk deney ve ilk kez
    # kullanilan her model haksiz yere yavas gorunur.
    pipeline = PerceptionPipeline(config)
    started = time.perf_counter()
    pipeline.analyze(video)
    elapsed = time.perf_counter() - started

    stats = pipeline.tracker.stats
    if not stats.frames_seen:
        return {"kimlik": 0, "medyan_iz": 0, "parcalanma": 0.0, "delikli": 0.0, "es_zamanli": 0.0, "fps": 0.0}

    lengths = list(stats.frames_seen.values())
    total = len(stats.frames_seen)
    return {
        "kimlik": total,
        "medyan_iz": statistics.median(lengths),
        "parcalanma": len(stats.fragmented(base.tracking.min_track_len)) / total * 100,
        "delikli": len(stats.gapped()) / total * 100,
        "es_zamanli": statistics.mean(stats.concurrent),
        "fps": len(stats.concurrent) / elapsed if elapsed > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None, help="Baz alinacak YAML")
    parser.add_argument("--only", type=str, default=None, help="Virgulle ayrilmis deney adlari")
    args = parser.parse_args(argv)

    if not args.video.is_file():
        raise SystemExit(f"Video bulunamadi: {args.video}")

    base = Config.load(args.config)
    selected = args.only.split(",") if args.only else list(EXPERIMENTS)
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Bilinmeyen deney: {', '.join(unknown)}. Secenekler: {', '.join(EXPERIMENTS)}")

    header = (
        f"{'deney':<14} | {'kimlik':>6} | {'medyan iz':>9} | "
        f"{'parcalanma':>10} | {'delikli':>7} | {'es zaman':>8} | {'FPS':>5}"
    )
    print(f"\n{args.video.name} uzerinde {len(selected)} deney\n")
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, float]] = {}
    for name in selected:
        description, det_over, trk_over = EXPERIMENTS[name]
        # Alt surecte degil ayni surecte kosuyoruz; her deney kendi
        # pipeline'ini kurdugu icin takip durumu tasinmiyor.
        row = run_one(args.video, base, det_over, trk_over)
        results[name] = row
        print(
            f"{name:<14} | {row['kimlik']:>6.0f} | {row['medyan_iz']:>9.0f} | "
            f"{row['parcalanma']:>9.0f}% | {row['delikli']:>6.0f}% | "
            f"{row['es_zamanli']:>8.1f} | {row['fps']:>5.1f}",
            flush=True,
        )

    print("\nAciklamalar:")
    for name in selected:
        print(f"  {name:<14} {EXPERIMENTS[name][0]}")

    if "baz" in results and len(results) > 1:
        best = min(
            (n for n in results if n != "baz"),
            key=lambda n: (results[n]["parcalanma"], results[n]["kimlik"]),
        )
        base_row, best_row = results["baz"], results[best]
        print(
            f"\nEn dusuk parcalanma: '{best}' "
            f"({base_row['parcalanma']:.0f}% -> {best_row['parcalanma']:.0f}%, "
            f"kimlik {base_row['kimlik']:.0f} -> {best_row['kimlik']:.0f}, "
            f"{base_row['fps']:.1f} -> {best_row['fps']:.1f} FPS)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
