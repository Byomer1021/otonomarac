"""Kusbakisi projeksiyon - Yontem A, homografi (Hafta 4).

Yol duz bir duzlem kabul edilir. Goruntudeki dort nokta
(`camera.road_quad`) zemin duzlemindeki bir dikdortgenin koseleri sayilir ve
aradaki perspektif donusum matrisi hesaplanir. Bir nesnenin kutusunun alt orta
noktasi - zemine degdigi varsayilan yer - bu matrisle harita koordinatina
cevrilir.

Yontem B (derinlik + kamera ic parametreleriyle 3B'ye tasima) ikinci asamada
eklenip ikisi karsilastirilacak. Bu dosya yalnizca A'yi uygular.

Koordinat sistemi (metre):

    x: aracin sagi pozitif, solu negatif
    y: aracin ilerisi pozitif

Ego aracin zemine degdigi nokta orijindir.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BEVConfig, CameraConfig
from .detection import Detection


@dataclass(frozen=True)
class GroundPoint:
    """Zemin duzlemindeki bir nokta (metre)."""

    x: float
    y: float

    @property
    def range_m(self) -> float:
        """Ego araca olan duz mesafe."""
        return float(np.hypot(self.x, self.y))


class BEVProjector:
    """Goruntu noktalarini zemin duzlemine yansitir ve harita cizer."""

    def __init__(self, config: BEVConfig | None = None, camera: CameraConfig | None = None) -> None:
        self.config = config or BEVConfig()
        self.camera = camera or CameraConfig()

        if self.camera.road_quad is None:
            raise ValueError(
                "camera.road_quad tanimli degil. Homografi icin yol duzleminde dort "
                "nokta gerekiyor - scripts/calibrate_bev.py ile secin."
            )
        if len(self.camera.road_quad) != 4:
            raise ValueError(f"road_quad 4 nokta olmali, {len(self.camera.road_quad)} verildi")

        #: Homografi ilk kare geldiginde kurulur; kaynak noktalar oran olarak
        #: saklandigi icin gercek piksel degerleri kare boyutuna bagli.
        self._H: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None

    # ---------- kurulum ----------

    def _ensure_homography(self, shape: tuple[int, int]) -> np.ndarray:
        if self._H is not None and self._frame_shape == shape:
            return self._H

        height, width = shape
        src = np.array(
            [[u * width, v * height] for u, v in self.camera.road_quad], dtype=np.float32
        )

        # Hedef: gercek dunyada bir dikdortgen. Genislik serit isaretinden
        # olculebiliyor, derinlik varsayim (bkz. BEVConfig.quad_depth_m).
        half_w = self.config.quad_width_m / 2
        depth = self.config.quad_depth_m
        # Kaynak sirasi: yakin-sol, yakin-sag, uzak-sag, uzak-sol
        dst = np.array(
            [
                [-half_w, 0.0],
                [half_w, 0.0],
                [half_w, depth],
                [-half_w, depth],
            ],
            dtype=np.float32,
        )

        self._H = cv2.getPerspectiveTransform(src, dst)
        self._frame_shape = shape
        return self._H

    # ---------- projeksiyon ----------

    def to_ground(self, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """(N,2) goruntu noktasini (N,2) zemin koordinatina (metre) cevirir."""
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        H = self._ensure_homography(shape)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def project(self, detections: list[Detection], shape: tuple[int, int]) -> list[Detection]:
        """Her tespitin zemine degme noktasini haritaya tasir (yerinde degistirir).

        Kaputun altina dusen temas noktalari gecersiz sayilir: o piksel yolu
        degil aracin kendi kaputunu gosterir, dolayisiyla homografi oradan
        anlamli bir zemin konumu uretemez.
        """
        if not detections:
            return detections

        height = shape[0]
        hood_row = height * self.camera.hood_top if self.camera.hood_top is not None else height

        usable: list[Detection] = []
        contacts: list[tuple[float, float]] = []
        for det in detections:
            cx, cy = det.bottom_center
            if cy > hood_row:
                det.bev_xy = None
                continue
            usable.append(det)
            contacts.append((cx, cy))

        if not usable:
            return detections

        ground = self.to_ground(np.array(contacts, dtype=np.float32), shape)
        for det, (gx, gy) in zip(usable, ground):
            # Arkada veya cok uzakta cikan noktalar atilir: homografi ufka
            # yaklastikca hizla bozulur, oradaki deger olculmus sayilmamali.
            if gy <= 0 or gy > self.config.max_range_m:
                det.bev_xy = None
            else:
                det.bev_xy = (float(gx), float(gy))

        return detections

    # ---------- cizim ----------

    @property
    def canvas_size(self) -> tuple[int, int]:
        """(genislik, yukseklik) piksel."""
        w = int(round(2 * self.config.range_side_m * self.config.px_per_m))
        h = int(round(self.config.range_ahead_m * self.config.px_per_m))
        return w, h

    def to_canvas(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Zemin koordinatini harita pikseline cevirir.

        Harita yukari dogru ilerlemeyi gosterir: ego arac altta ortada.
        """
        w, h = self.canvas_size
        px = w / 2 + x_m * self.config.px_per_m
        py = h - y_m * self.config.px_per_m
        return int(round(px)), int(round(py))

    def render(self, detections: list[Detection]) -> np.ndarray:
        """Kusbakisi haritayi cizer.

        Cizim burada, `visualize.py`'de degil: harita icerigi projeksiyon
        geometrisine sikica bagli (metre-piksel donusumu, menzil halkalari,
        ego konumu) ve ikisini ayirmak her cagride yarim duzine parametre
        tasimayi gerektirirdi.
        """
        from .visualize import class_color  # dairesel import olmasin diye burada

        w, h = self.canvas_size
        canvas = np.full((h, w, 3), 22, dtype=np.uint8)

        self._draw_grid(canvas)
        self._draw_ego(canvas, self.ego_offset_m)

        for det in detections:
            if det.bev_xy is None:
                continue
            x_m, y_m = det.bev_xy
            px, py = self.to_canvas(x_m, y_m)
            if not (0 <= px < w and 0 <= py < h):
                continue

            color = class_color(det.cls_id)
            # Uzaktaki nesne kucuk cizilir: hem okunurluk hem de projeksiyonun
            # orada daha belirsiz oldugunu ima etmek icin.
            radius = max(3, int(round(9 - y_m / self.config.range_ahead_m * 5)))
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius, (12, 12, 12), 1)

            if det.track_id is not None:
                cv2.putText(
                    canvas,
                    str(det.track_id),
                    (px + radius + 3, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def _draw_grid(self, canvas: np.ndarray) -> None:
        """Mesafe halkalari ve serit genisliginde dikey cizgiler."""
        h, w = canvas.shape[:2]
        line = (46, 46, 52)
        text = (120, 120, 130)

        step = self.config.grid_step_m
        distance = step
        while distance <= self.config.range_ahead_m:
            _, py = self.to_canvas(0.0, distance)
            cv2.line(canvas, (0, py), (w, py), line, 1)
            cv2.putText(
                canvas, f"{distance:.0f}m", (4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, text, 1, cv2.LINE_AA,
            )
            distance += step

        # Serit genisliginde dikey referanslar: haritadaki yanal olcegi
        # gozle dogrulamayi mumkun kilar.
        lane = self.config.quad_width_m
        offset = lane / 2
        while offset <= self.config.range_side_m:
            for sign in (-1, 1):
                px, _ = self.to_canvas(sign * offset, 0.0)
                cv2.line(canvas, (px, 0), (px, h), line, 1)
            offset += lane

    @property
    def ego_offset_m(self) -> float:
        """Kameranin zemin duzlemindeki yanal konumu (metre).

        Koordinat orijini, kalibrasyon dortgeninin yakin kenarinin ORTASI -
        yani seridin ortasi, aracin konumu degil. Kamera bu noktanin tam
        uzerinde olmak zorunda degil; gercek konumu, goruntunun orta
        sutununun yakin satirdaki zemin karsiligidir.

        Boylamsal konum (ego'nun y'si) ayni sekilde bilinmiyor ve bilinemez:
        dortgenin yakin kenarinin araca uzakligi olculmedi. Harita bu yuzden
        "yakin referans satirindan itibaren" mesafe gosteriyor.
        """
        if self._frame_shape is None:
            return 0.0
        height, width = self._frame_shape
        near_row = max(v for _, v in self.camera.road_quad) * height
        ground = self.to_ground(
            np.array([[width / 2, near_row]], dtype=np.float32), self._frame_shape
        )
        return float(ground[0][0])

    def _draw_ego(self, canvas: np.ndarray, offset_m: float = 0.0) -> None:
        """Ego araci haritanin altina, gercek yanal konumuna yerlestirir."""
        h, w = canvas.shape[:2]
        half = max(2, int(round(0.9 * self.config.px_per_m)))   # ~1.8 m genislik
        length = max(3, int(round(4.2 * self.config.px_per_m))) # ~4.2 m boy
        cx, _ = self.to_canvas(offset_m, 0.0)
        cv2.rectangle(canvas, (cx - half, h - length), (cx + half, h - 1), (200, 200, 205), -1)
        cv2.rectangle(canvas, (cx - half, h - length), (cx + half, h - 1), (60, 60, 66), 1)
        # Aracin ilerleme ekseni: yanal konumun dogru olup olmadigini gozle
        # dogrulamak icin referans.
        cv2.line(canvas, (cx, 0), (cx, h - length), (70, 70, 78), 1, cv2.LINE_AA)
