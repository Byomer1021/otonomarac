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

    def _depth() -> str:
        from perception.config import CameraConfig, DepthConfig
        from perception.depth import DepthEstimator, fuse
        from perception.detection import Detection

        from perception.depth import normalize_for_display

        estimator = DepthEstimator(DepthConfig(input_width=252))
        # Sentetik gurultu yerine dikey gradyan: gercek bir sahne gibi
        # ust-alt derinlik farki iceriyor, model anlamli bir harita uretebiliyor.
        frame = np.tile(
            np.linspace(20, 235, 180, dtype=np.uint8).reshape(-1, 1, 1), (1, 320, 3)
        ).astype(np.uint8)
        disparity = estimator.infer(frame)

        if disparity.shape != frame.shape[:2]:
            raise AssertionError(f"harita boyutu kareyle uyusmuyor: {disparity.shape} vs {frame.shape[:2]}")

        shown = normalize_for_display(disparity)
        if not (0.0 <= shown.min() and shown.max() <= 1.0):
            raise AssertionError(f"cizim normalizasyonu [0,1] disinda: {shown.min()}..{shown.max()}")

        det = Detection(xyxy=(80, 40, 200, 150), conf=0.9, cls_id=2)
        fuse([det], disparity, DepthConfig(), CameraConfig())

        # Kaput sinirinin tamamen altinda kalan kutu derinlik almamali.
        below = Detection(xyxy=(80, 170, 200, 180), conf=0.9, cls_id=2)
        fuse([below], disparity, DepthConfig(), CameraConfig(hood_top=0.5))
        if below.depth is not None:
            raise AssertionError("kaput altindaki kutu icin derinlik uretildi")

        shown_depth = f"~{det.depth:.1f}" if det.depth is not None else "gecersiz (beklenebilir)"
        return f"cihaz={estimator.device}, ham aralik {disparity.min():.2f}..{disparity.max():.2f}, ornek {shown_depth}"

    def _bev() -> str:
        from perception.bev import BEVProjector
        from perception.config import BEVConfig, CameraConfig
        from perception.detection import Detection

        # Yol duzleminde makul bir yamuk: yakin kenar genis, uzak kenar dar.
        camera = CameraConfig(
            hood_top=0.85,
            road_quad=[[0.41, 0.84], [0.65, 0.84], [0.58, 0.73], [0.49, 0.73]],
        )
        projector = BEVProjector(BEVConfig(quad_width_m=3.8, quad_depth_m=12.0), camera)
        shape = (720, 1280)

        near = Detection(xyxy=(600, 520, 700, 600), conf=0.9, cls_id=2, track_id=1)
        far = Detection(xyxy=(630, 505, 670, 530), conf=0.9, cls_id=2, track_id=2)
        projector.project([near, far], shape)

        if near.bev_xy is None or far.bev_xy is None:
            raise AssertionError(f"projeksiyon uretilemedi: {near.bev_xy}, {far.bev_xy}")
        if not far.bev_xy[1] > near.bev_xy[1]:
            raise AssertionError(
                f"uzaktaki nesne daha yakin cikti: {far.bev_xy[1]:.1f} <= {near.bev_xy[1]:.1f}"
            )

        # Kaputun altina dusen temas noktasi gecersiz olmali.
        under = Detection(xyxy=(600, 640, 700, 700), conf=0.9, cls_id=2)
        projector.project([under], shape)
        if under.bev_xy is not None:
            raise AssertionError("kaput altindaki kutu icin zemin konumu uretildi")

        canvas = projector.render([near, far])
        w, h = projector.canvas_size
        if canvas.shape[:2] != (h, w):
            raise AssertionError(f"harita boyutu yanlis: {canvas.shape[:2]} != {(h, w)}")

        return f"yakin {near.bev_xy[1]:.1f}m < uzak {far.bev_xy[1]:.1f}m, harita {w}x{h}"

    def _risk() -> str:
        from perception.config import RiskConfig
        from perception.detection import Detection
        from perception.risk import RiskEstimator

        estimator = RiskEstimator(RiskConfig())
        # Sabit hizla yaklasan tek nesne: 20 m'den 2 m/s ile.
        ttc = None
        for step in range(12):
            det = Detection(xyxy=(600, 400, 700, 500), conf=0.9, cls_id=2, track_id=1)
            det.bev_xy = (0.0, 20.0 - step * 2.0 * 0.1)
            estimator.update([det], timestamp=step * 0.1)
            ttc = det.ttc

        if ttc is None:
            raise AssertionError("duzgun yaklasan nesne icin TTC uretilmedi")
        # 20 m'den 2 m/s: son adimda mesafe ~17.8 m, TTC ~8.9 s beklenir.
        if not 7.0 < ttc < 11.0:
            raise AssertionError(f"TTC beklenen araligin disinda: {ttc:.1f}s")

        # Koridor disindaki nesne risk almamali.
        side = Detection(xyxy=(600, 400, 700, 500), conf=0.9, cls_id=2, track_id=2)
        side.bev_xy = (6.0, 10.0)
        estimator.update([side], timestamp=2.0)
        if side.ttc is not None:
            raise AssertionError("guzergah disindaki nesne icin TTC uretildi")

        return f"yaklasan nesne TTC {ttc:.1f}s, koridor disi elendi"

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
        ("Depth Anything + fuzyon", _depth),
        ("Kusbakisi projeksiyon", _bev),
        ("Hiz ve TTC", _risk),
        ("Cizim", _draw),
    ]:
        results.append(check(name, fn))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} kontrol gecti")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
