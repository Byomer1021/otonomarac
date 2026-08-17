"""Kusbakisi homografisi icin dort nokta secer ve sonucu DOGRULAR (Hafta 4).

Homografiyi kurmak kolay; dogru kurdugunu bilmek zor. Bu betik iki is yapiyor:

1. Secilen dortgeni kare uzerine cizip yol duzlemini yukaridan gosteren
   duzlestirilmis bir gorsel uretir - noktalar gozle ayarlanabilir.
2. Kalibrasyonu **olcer**: yolda gercekte paralel olan bir cizgi (serit boyasi)
   duzlestirilmis gorunumde ne kadar dikey duruyor? Perspektif dogru
   kaldirildiysa o cizginin genisligi derinlikle degismemeli.

Kullanim:
    python scripts/calibrate_bev.py data/maltepe_city.mp4 --frame 900
    python scripts/calibrate_bev.py data/maltepe_city.mp4 --frame 900 \
        --quad 0.417,0.833 0.680,0.833 0.556,0.722 0.497,0.722
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception.config import Config  # noqa: E402

#: Sarı serit boyasini yakalayan HSV araligi. Kalibrasyon dogrulamasinda
#: "gercekte duz olan cizgi" olarak bu kullaniliyor.
YELLOW_LO, YELLOW_HI = (15, 80, 110), (38, 255, 255)

#: Dogrulamada kullanilan tespit guven esigi. Config'deki degerden bagimsiz
#: ve belirgin yuksek - gerekcesi verify_with_vehicles icinde.
VERIFY_CONF = 0.40


def load_frame(video: Path, index: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{index}. kare okunamadi: {video}")
    if width and frame.shape[1] != width:
        scale = width / frame.shape[1]
        frame = cv2.resize(
            frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    return frame


def warp_road(
    frame: np.ndarray,
    quad: list[list[float]],
    out_size: tuple[int, int],
    margin: float = 0.35,
) -> np.ndarray:
    """Yol duzlemini yukaridan bakan bir gorsele duzlestirir.

    Dortgen ciktinin tamamina degil, `margin` oraninda kenar payi birakilarak
    ortasina yerlestirilir. Boylece dortgenin kenarina denk gelen serit boyasi
    kirpilmadan gorunur ve dogrulama olcumu onu bulabilir.
    """
    h, w = frame.shape[:2]
    src = np.array([[u * w, v * h] for u, v in quad], dtype=np.float32)

    ow, oh = out_size
    mx, my = ow * margin, oh * margin * 0.5
    dst = np.array(
        [[mx, oh - my], [ow - mx, oh - my], [ow - mx, my], [mx, my]], dtype=np.float32
    )
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (ow, oh))


def verify_with_vehicles(
    video: Path, config: Config, quad: list[list[float]], width: int, samples: int
) -> tuple[float, float, int] | None:
    """Kalibrasyonu BAGIMSIZ bir referansla olcer: arac genisligi.

    Ilk surumde dogrulama, dortgenin sol kenari olarak kullanilan sari
    cizginin duzlestirilmis gorunumde dikey cikip cikmadigina bakiyordu. O
    olcum DONGUSELDI - dortgen zaten o cizgiden kuruluyordu, dolayisiyla
    sonuc her zaman mukemmel geliyordu ve onlarca farkli ufuk degeri ayni
    skoru aliyordu, yani kalibrasyonu hic kisitlamiyordu.

    Bunun yerine kalibrasyonda hic kullanilmayan bir buyukluk olculuyor:
    tespit edilen araclarin zemine yansitilmis genisligi. Iki sey beklenir -
    medyan tipik arac genisligine (~1.8 m) yakin olmali, ve genislik mesafeden
    BAGIMSIZ olmali. Ikincisi asil testtir: korelasyon sifirdan uzaksa
    perspektif dogru kaldirilmamis demektir.

    Yalnizca one yakin (yanal olarak merkeze yakin) araclar sayilir; acili
    gorunen bir aracin kutusu gercek genisliginden genistir.

    Returns:
        (medyan_genislik_m, mesafe_korelasyonu, ornek_sayisi) ya da None.
    """
    from dataclasses import replace

    from perception.bev import BEVProjector
    from perception.config import CameraConfig
    from perception.detection import VEHICLE_CLASSES, YOLODetector

    camera = CameraConfig(hood_top=config.camera.hood_top, road_quad=quad)
    projector = BEVProjector(config.bev, camera)
    # Config'deki guven esigi takip icin bilincli olarak dusuk tutuluyor
    # (ByteTrack'in ikinci eslestirme turu icin). Burada isimize yaramaz:
    # kismi ve ortulu kutular gercekte olduklarindan dar olcusur ve medyani
    # asagi ceker. Dogrulama guvenilir kutu ister, yuksek geri cagirim degil.
    detector = YOLODetector(replace(config.detection, conf=VERIFY_CONF))

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or samples * 100
    step = max(1, total // samples)

    widths, ranges = [], []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width:
            scale = width / frame.shape[1]
            frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        shape = frame.shape[:2]
        hood = shape[0] * (config.camera.hood_top or 1.0)

        for det in detector.detect(frame):
            if det.cls_id not in VEHICLE_CLASSES:
                continue
            x1, _, x2, y2 = det.xyxy
            if y2 > hood:
                continue
            ground = projector.to_ground(np.array([[x1, y2], [x2, y2]], np.float32), shape)
            near, far = ground[0][1], ground[1][1]
            if min(near, far) <= 1 or max(near, far) > config.bev.max_range_m:
                continue
            centre_x = (ground[0][0] + ground[1][0]) / 2
            if abs(centre_x) > 3.5:
                continue
            widths.append(abs(ground[1][0] - ground[0][0]))
            ranges.append((near + far) / 2)
    cap.release()

    if len(widths) < 8:
        return None
    return float(np.median(widths)), float(np.corrcoef(ranges, widths)[0, 1]), len(widths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=900)
    parser.add_argument("--width", type=int, default=1280, help="Isleme genisligi")
    parser.add_argument(
        "--quad",
        nargs=4,
        default=None,
        metavar="U,V",
        help="Dort nokta (oran): yakin-sol yakin-sag uzak-sag uzak-sol",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/bev_calib.png"))
    parser.add_argument("--verify", action="store_true", help="Arac genisligiyle bagimsiz dogrulama")
    parser.add_argument("--samples", type=int, default=15, help="Dogrulamada ornekleneek kare sayisi")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    if args.quad:
        quad = [[float(v) for v in p.split(",")] for p in args.quad]
    elif config.camera.road_quad:
        quad = config.camera.road_quad
    else:
        raise SystemExit("Dortgen yok: --quad verin ya da config'e camera.road_quad yazin.")

    frame = load_frame(args.video, args.frame, args.width)
    h, w = frame.shape[:2]
    px = [(int(u * w), int(v * h)) for u, v in quad]

    warped = warp_road(frame, quad, (420, 620))

    print(f"\nkare {args.frame}, {w}x{h}")
    print("dortgen (piksel):", " ".join(f"({x},{y})" for x, y in px))

    if args.verify:
        print(f"\n{args.samples} kare uzerinde arac genisligi olculuyor...", flush=True)
        result = verify_with_vehicles(args.video, config, quad, args.width, args.samples)
        if result is None:
            print("Dogrulanamadi: yeterli sayida uygun arac bulunamadi.")
        else:
            median_w, corr, n = result
            print(f"\n  ornek sayisi                   : {n}")
            print(f"  medyan yansitilmis genislik    : {median_w:5.2f} m    (tipik arac ~1.80)")
            print(f"  genisligin mesafeyle korelasyonu: {corr:+5.2f}       (hedef 0)")
            verdict = (
                "iyi" if abs(corr) < 0.15 and 1.5 < median_w < 2.2
                else "kabul edilebilir" if abs(corr) < 0.30
                else "zayif - ufuk yuksekligini ayarlayin"
            )
            print(f"  degerlendirme                  : {verdict}")
            if abs(median_w - 1.80) > 0.1:
                print(f"\n  Oneri: quad_width_m *= {1.80 / median_w:.3f}"
                      f"  ({config.bev.quad_width_m:.2f} -> {config.bev.quad_width_m * 1.80 / median_w:.2f})")
        print("\n  Not: quad_depth_m bu testle kisitlanmaz - degistirildiginde")
        print("  genislik de korelasyon da degismez, sadece mutlak mesafeler")
        print("  olceklenir. Boylamsal olcek Hafta 5'e kaldi.")

    overlay = frame.copy()
    cv2.polylines(overlay, [np.array(px, np.int32)], True, (0, 230, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(px):
        cv2.circle(overlay, (x, y), 6, (0, 0, 235), -1)
        cv2.putText(overlay, "yaSL yaSA uzSA uzSO".split()[i], (x + 9, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 235), 1, cv2.LINE_AA)
    if config.camera.hood_top:
        row = int(config.camera.hood_top * h)
        cv2.line(overlay, (0, row), (w, row), (255, 0, 255), 2)

    panel = cv2.resize(warped, (420, overlay.shape[0]), interpolation=cv2.INTER_AREA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.hstack([overlay, panel]))
    print(f"\ngorsel: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
