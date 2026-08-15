"""KITTI raw 'sync' zip dosyasindan islenebilir bir mp4 uretir.

KITTI kareleri tek tek PNG olarak dagitir ve zip icinde LiDAR, IMU, dort ayri
kamera da gelir. Pipeline'in ihtiyaci olan tek sey sol renkli kamera (image_02),
bu yuzden zip acilmadan dogrudan icinden okunur - diske 600 MB'lik ikinci bir
kopya cikarmaya gerek yok.

Kullanim:
    python scripts/kitti_to_video.py data/kitti/drive_0005.zip -o data/kitti_0005.mp4
    python scripts/kitti_to_video.py data/kitti/calib.zip --extract-calib data/kitti/calib
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

#: KITTI raw veri 10 Hz kaydedilmistir.
KITTI_FPS = 10.0

#: Kamera klasoru -> aciklama. image_02 = sol renkli kamera, projemizin girdisi.
CAMERA_DIRS = {
    "image_00": "sol gri",
    "image_01": "sag gri",
    "image_02": "sol renkli",
    "image_03": "sag renkli",
}


def frame_members(archive: zipfile.ZipFile, camera: str) -> list[str]:
    """Zip icindeki kare PNG'lerini dosya adina gore sirali dondurur."""
    needle = f"/{camera}/data/"
    members = [
        name
        for name in archive.namelist()
        if needle in name and name.lower().endswith(".png")
    ]
    # KITTI dosya adlari sifir dolgulu (0000000000.png), bu yuzden metin
    # siralamasi kare sirasiyla ayni.
    return sorted(members)


def convert(zip_path: Path, output: Path, camera: str, fps: float, max_frames: int | None) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = frame_members(archive, camera)
        if not members:
            available = sorted({d for d in CAMERA_DIRS if any(f"/{d}/" in n for n in archive.namelist())})
            raise SystemExit(
                f"'{camera}' klasorunde kare bulunamadi. Zip'te olanlar: {available or 'yok'}"
            )

        if max_frames is not None:
            members = members[:max_frames]

        output.parent.mkdir(parents=True, exist_ok=True)
        writer: cv2.VideoWriter | None = None

        try:
            for i, member in enumerate(members):
                buffer = np.frombuffer(archive.read(member), dtype=np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"  uyari: cozulemedi, atlandi -> {member}", file=sys.stderr)
                    continue

                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise SystemExit(f"Cikti videosu acilamadi: {output}")
                    print(f"  {len(members)} kare, {width}x{height} @ {fps} FPS")

                writer.write(image)
                if (i + 1) % 25 == 0 or i + 1 == len(members):
                    print(f"  {i + 1}/{len(members)}", end="\r")
        finally:
            if writer is not None:
                writer.release()

    print()
    return output


def extract_calib(zip_path: Path, target: Path) -> None:
    """Kalibrasyon .txt dosyalarini duz bir klasore cikarir."""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".txt")]
        if not names:
            raise SystemExit(f"Zip icinde .txt kalibrasyon dosyasi yok: {zip_path}")
        for name in names:
            with archive.open(name) as src, (target / Path(name).name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  {Path(name).name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip_path", type=Path, help="KITTI *_sync.zip veya calib.zip")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Cikti mp4 yolu")
    parser.add_argument("--camera", default="image_02", choices=sorted(CAMERA_DIRS), help="Kamera klasoru")
    parser.add_argument("--fps", type=float, default=KITTI_FPS, help="Cikti FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Sadece ilk N kare")
    parser.add_argument("--extract-calib", type=Path, default=None, help="Kalibrasyonu bu klasore cikar")
    args = parser.parse_args(argv)

    if not args.zip_path.is_file():
        raise SystemExit(f"Zip bulunamadi: {args.zip_path}")

    if args.extract_calib is not None:
        print(f"Kalibrasyon cikariliyor -> {args.extract_calib}")
        extract_calib(args.zip_path, args.extract_calib)
        return 0

    output = args.output or args.zip_path.with_suffix(".mp4")
    print(f"{args.zip_path.name} ({CAMERA_DIRS[args.camera]}) -> {output}")
    convert(args.zip_path, output, args.camera, args.fps, args.max_frames)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Bitti: {output.resolve()} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
