"""Sistemin nerede bozuldugunu OLCER (Hafta 8).

On rapor hata analizini basari kriteri sayiyor: README'de sistemin ne yaptigi,
nasil calistigi ve NEREDE BASARISIZ OLDUGU yazili olmali. Bu betik o bolumun
rakamlarini uretiyor - iddia degil olcum.

Dort olcum:

1. Bilgi hunisi. Ham tespitten TTC'ye kadar her adimda kac nesne eleniyor ve
   neden. Sistemin neyi kaybettigi, neyi urettigi kadar onemli.

2. Mesafe kararliligi. Ayni iz icin ardisik kareler arasindaki mesafe degisim
   hizi. Duran bir nesnede bu deger olcum gurultusudur; mesafe bandlarina
   ayirinca homografinin uzakta ne kadar bozuldugu gorunur.

3. Ortulme etkisi. Kutusu daha yakin bir kutuyla ortusen tespitlerin olcumu,
   ortusmeyenlerle karsilastiriliyor. Hafta 5'te tekil bir ornekte gorulen
   sorun burada sayiya donusuyor.

4. Sahne bagimliligi. Sehir ici ve kirsal klipler yan yana.

Kullanim:
    python scripts/failure_analysis.py
    python scripts/failure_analysis.py --frames 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception.config import Config  # noqa: E402
from perception.detection import VEHICLE_CLASSES  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402
from perception.video_io import VideoReader  # noqa: E402

#: Mesafe kararliligi bu bandlarda raporlaniyor (metre).
BANDS = [(0, 8), (8, 15), (15, 25), (25, 40)]

#: Bir kutunun ortulu sayilmasi icin daha yakin bir kutuyla asgari ortusme.
OCCLUSION_IOU = 0.15

#: Sahne tablosu: ad -> (dosya, hood_top, kusbakisi_acik_mi).
#: hood_top sahneye ozel cunku yagmur kaydi baska bir kadrajdan geliyor -
#: kaput orada goruntunun yalnizca alt %3'unu kapliyor, Maltepe'de %15.
#: Kusbakisi ve risk yalnizca kalibrasyonu olculmus kamerada acik; yagmur
#: kaydinin homografisi cikarilmadi, o yuzden orada tespit ve takip olculuyor.
SCENES = {
    "kuru sehir": ("data/maltepe_city.mp4", 0.85, True),
    "kuru kirsal": ("data/maltepe_rural.mp4", 0.85, True),
    "yagmur yol": ("data/rain_highway.mp4", 0.96, False),
    "yagmur sis": ("data/rain_fog.mp4", 0.96, False),
}

FUNNEL = [
    ("ham_tespit", "ham tespit"),
    ("kimlik_alan", "kimlik atandi"),
    ("derinlik_alan", "derinlik uretildi"),
    ("zemin_konumu", "zemin konumu var"),
    ("ttc_alan", "TTC uretildi"),
]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def analyse(video: Path, config: Config, frames: int) -> dict:
    pipeline = PerceptionPipeline(config)
    counts = defaultdict(int)
    #: iz -> [(zaman, mesafe, ortulu_mu)]
    history: dict[int, list[tuple[float, float, bool]]] = defaultdict(list)

    with VideoReader(video, max_frames=frames, resize_width=config.video.resize_width) as reader:
        if pipeline.tracker is not None:
            pipeline.tracker.configure_for_fps(reader.output_meta.fps)

        for frame in reader:
            result = pipeline.process_frame(frame)
            objects = result.objects

            counts["ham_tespit"] += len(result.detections)
            counts["kimlik_alan"] += len(objects)
            counts["derinlik_alan"] += sum(1 for d in objects if d.depth is not None)
            counts["zemin_konumu"] += sum(1 for d in objects if d.bev_xy is not None)
            counts["ttc_alan"] += sum(1 for d in objects if d.ttc is not None)

            hood = frame.image.shape[0] * (config.camera.hood_top or 1.0)
            counts["kaput_altinda"] += sum(
                1 for d in objects if d.bev_xy is None and d.bottom_center[1] > hood
            )

            vehicles = [d for d in objects if d.cls_id in VEHICLE_CLASSES and d.bev_xy]
            for det in vehicles:
                nearer = [o for o in vehicles if o is not det and o.bev_xy[1] < det.bev_xy[1]]
                occluded = any(_iou(det.xyxy, o.xyxy) > OCCLUSION_IOU for o in nearer)
                history[det.track_id].append((frame.timestamp, det.bev_xy[1], occluded))

    return {"counts": counts, "history": history, "pipeline": pipeline}


def _rates(history: dict[int, list]) -> tuple[list, list]:
    """Ardisik gozlemler arasi mesafe degisim hizi (m/s), ortusuz ve ortulu."""
    clear, occluded = [], []
    for samples in history.values():
        for (t0, d0, o0), (t1, d1, o1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if not 0 < dt < 0.5:
                continue
            rate = abs(d1 - d0) / dt
            (occluded if (o0 or o1) else clear).append((d1, rate))
    return clear, occluded


def _summarise(values: list[float]) -> tuple[int, float, float] | None:
    vals = sorted(values)
    if len(vals) < 10:
        return None
    return len(vals), statistics.median(vals), vals[int(len(vals) * 0.9)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/maltepe.yaml"))
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument(
        "--only", type=str, default=None,
        help="Virgulle ayrilmis sahne adlari: " + ", ".join(SCENES),
    )
    args = parser.parse_args(argv)

    base = Config.load(args.config)
    scenes = {}
    selected = args.only.split(",") if args.only else list(SCENES)
    for label in selected:
        if label not in SCENES:
            raise SystemExit(f"Bilinmeyen sahne: {label}. Secenekler: {', '.join(SCENES)}")
        name, hood, use_bev = SCENES[label]
        path = Path(name)
        if not path.is_file():
            print(f"atlaniyor - dosya yok: {path}")
            continue
        cfg = Config()
        cfg.video = replace(base.video, input=str(path), max_frames=args.frames)
        cfg.camera = replace(
            base.camera, hood_top=hood,
            road_quad=base.camera.road_quad if use_bev else None,
        )
        cfg.detection = replace(base.detection)
        cfg.tracking = replace(base.tracking)
        # Derinlik ve segmentasyon bu analizin sonuclarini etkilemiyor ve
        # CPU'da en pahali iki katman; kapatilinca olcum dakikalar yerine
        # saniyeler suruyor.
        cfg.depth = replace(base.depth, enabled=False)
        cfg.segmentation = replace(base.segmentation, enabled=False)
        cfg.bev = replace(base.bev, enabled=use_bev)
        cfg.risk = replace(base.risk, enabled=use_bev)
        cfg.visualize = replace(base.visualize)
        print(f"{label} isleniyor ({path.name}, {args.frames} kare)...", flush=True)
        scenes[label] = analyse(path, cfg, args.frames)

    if not scenes:
        raise SystemExit("Hicbir klip bulunamadi.")

    print("\n\n1. BILGI HUNISI  (gozlem sayisi ve ham tespite orani)")
    header = f"{'adim':<20}" + "".join(f"{s:>17}" for s in scenes)
    print(header)
    print("-" * len(header))
    for key, name in FUNNEL:
        row = f"{name:<20}"
        for s in scenes.values():
            c = s["counts"]
            pct = c[key] / c["ham_tespit"] * 100 if c["ham_tespit"] else 0
            row += f"{c[key]:>10} ({pct:>3.0f}%)"
        print(row)
    row = f"{'  kaput altinda':<20}"
    for s in scenes.values():
        row += f"{s['counts']['kaput_altinda']:>17}"
    print(row)

    print("\n\n2. MESAFE KARARLILIGI  (ardisik gozlemler arasi degisim, m/s)")
    print("   Duran bir nesnede bu deger olcum gurultusudur; buyumesi")
    print("   projeksiyonun o mesafede guvenilmez oldugunu gosterir.\n")
    for label, s in scenes.items():
        clear, occ = _rates(s["history"])
        print(f"  {label}")
        print(f"    {'band':<10} {'ornek':>7} {'medyan':>9} {'90. yuzdelik':>14}")
        for lo, hi in BANDS:
            vals = [r for d, r in clear + occ if lo <= d < hi]
            summary = _summarise(vals)
            if summary is None:
                print(f"    {f'{lo}-{hi} m':<10} {len(vals):>7}   (yetersiz ornek)")
            else:
                n, med, p90 = summary
                print(f"    {f'{lo}-{hi} m':<10} {n:>7} {med:>8.1f} {p90:>13.1f}")
        print()

    print("\n3. ORTULME ETKISI  (ayni olcut, ortulu ve ortusuz kutular)")
    print("   Ortulu aracin zemine degme noktasi gorunmez; kutunun gorunen alt")
    print("   kenari gercek temas noktasinin ustunde kalir ve homografi araci")
    print("   oldugundan uzaga koyar.\n")
    print(f"  {'sahne':<12} {'durum':<10} {'ornek':>7} {'medyan':>9} {'90. yuzdelik':>14}")
    for label, s in scenes.items():
        clear, occ = _rates(s["history"])
        for name, pairs in (("ortusuz", clear), ("ortulu", occ)):
            summary = _summarise([r for _, r in pairs])
            if summary is None:
                print(f"  {label:<12} {name:<10} {len(pairs):>7}   (yetersiz ornek)")
            else:
                n, med, p90 = summary
                print(f"  {label:<12} {name:<10} {n:>7} {med:>8.1f} {p90:>13.1f}")

    print("\n\n4. TAKIP VE RISK  (sahne bagimliligi)")
    for label, s in scenes.items():
        pipeline = s["pipeline"]
        print(f"\n  {label}")
        report = pipeline.tracker.stats.format_report(base.tracking.min_track_len)
        print("   " + report.replace("\n", "\n   "))
        if pipeline.risk is not None:
            print("   " + pipeline.risk.summary().replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
