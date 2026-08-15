"""Kurulumun saglam oldugunu dogrulayan hizli kontrol.

Video gerektirmez; sentetik bir kare uzerinde tum katmanlari bir kez calistirir.
Yeni bir makinede (ya da Colab'da) ilk yapilacak sey budur:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np


def check(name: str, fn) -> bool:
    try:
        detail = fn()
    except Exception:
        print(f"  [HATA] {name}")
        traceback.print_exc()
        return False
    print(f"  [OK]   {name}" + (f" - {detail}" if detail else ""))
    return True


def main() -> int:
    print("Kurulum kontrolu\n")
    results: list[bool] = []

    def _torch() -> str:
        import torch

        if not torch.cuda.is_available():
            return f"torch {torch.__version__} (CPU - CUDA gorunmuyor)"
        name = torch.cuda.get_device_name(0)
        capability = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
        return f"torch {torch.__version__} | {name} | sm_{capability.replace('.', '')}"

    def _cuda_matmul() -> str:
        import torch

        if not torch.cuda.is_available():
            return "atlandi (CUDA yok)"
        # Pascal kartlarda yanlis CUDA yapisi kurulmussa hata tam burada patlar.
        a = torch.randn(256, 256, device="cuda")
        return f"GPU matmul calisti ({(a @ a).sum().item():.1f})"

    def _opencv() -> str:
        import cv2

        return f"opencv {cv2.__version__}"

    def _package() -> str:
        from perception import __version__
        from perception.config import Config

        Config()
        return f"perception {__version__}"

    def _detector() -> str:
        from perception.config import DetectionConfig
        from perception.detection import YOLODetector

        detector = YOLODetector(DetectionConfig(imgsz=320))
        # Sentetik gri kare: tespit cikmasi beklenmiyor, amac hattin patlamamasi.
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        detections = detector.detect(frame)
        return f"cihaz={detector.device}, {len(detections)} tespit (sentetik karede 0 normal)"

    def _tracker() -> str:
        from perception.detection import Detection
        from perception.tracking import ObjectTracker

        tracker = ObjectTracker()
        # Sabit hizla saga kayan tek bir kutu: iki karede de ayni kimligi
        # almazsa takip katmani bozuk demektir.
        ids = []
        for step in range(4):
            det = Detection(xyxy=(100 + step * 8, 200, 180 + step * 8, 300), conf=0.9, cls_id=2)
            tracked = tracker.update([det], frame_index=step)
            ids.extend(d.track_id for d in tracked)

        unique = set(ids)
        if len(unique) != 1:
            raise AssertionError(f"tek nesne icin {len(unique)} kimlik uretildi: {sorted(unique)}")

        # Iz gecmisi de birikmis olmali, yoksa cizim katmani bos kalir.
        trail = tracker.trail_for(unique.copy().pop())
        if len(trail) < 2:
            raise AssertionError(f"iz gecmisi birikmedi: {trail}")
        return f"4 karede kimlik korundu (#{unique.pop()}), iz {len(trail)} nokta"

    def _draw() -> str:
        from perception.detection import Detection
        from perception.visualize import draw_detections, draw_hud, draw_trails

        canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        # Iz noktalari (kare_no, x, y) - kare numarasi bosluk tespiti icin.
        draw_trails(canvas, {7: [(0, 70.0, 200.0), (1, 68.0, 195.0), (2, 65.0, 188.0)]}, {7: 2})
        draw_detections(canvas, [Detection(xyxy=(20, 30, 120, 200), conf=0.9, cls_id=2, track_id=7)])
        draw_hud(canvas, ["30.0 FPS", "Aktif iz: 1"])
        return "kutu + iz + HUD cizildi"

    for name, fn in [
        ("PyTorch", _torch),
        ("CUDA calisma testi", _cuda_matmul),
        ("OpenCV", _opencv),
        ("perception paketi", _package),
        ("YOLO tespit", _detector),
        ("ByteTrack takip", _tracker),
        ("Cizim", _draw),
    ]:
        results.append(check(name, fn))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} kontrol gecti")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
