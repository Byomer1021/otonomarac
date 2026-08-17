"""Cizim katmani.

Tum OpenCV cizim islemleri burada toplanir. Renkler sinif bazinda sabittir -
video boyunca ayni sinif hep ayni renkte gorunur, bu da cikti GIF'inin
okunabilirligini ciddi bicimde artirir.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import VisualizeConfig
from .detection import Detection

#: Sinif -> BGR rengi. Kirilgan yol kullanicilari sicak, araclar soguk tonlarda.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 140, 255),    # yaya       - turuncu
    1: (0, 200, 255),    # bisiklet   - amber
    2: (80, 200, 80),    # araba      - yesil
    3: (200, 160, 60),   # motosiklet - camgobegi
    5: (200, 100, 60),   # otobus     - mavi
    7: (160, 80, 160),   # kamyon     - mor
    9: (60, 60, 220),    # trafik isigi - kirmizi
}
_DEFAULT_COLOR = (200, 200, 200)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(cls_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS.get(cls_id, _DEFAULT_COLOR)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    config: VisualizeConfig | None = None,
) -> np.ndarray:
    """Tespit kutularini ve etiketlerini kare uzerine cizer (yerinde degistirir)."""
    cfg = config or VisualizeConfig()

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        color = class_color(det.cls_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, cfg.line_thickness)
        label = _build_label(det, cfg)
        if label:
            _draw_label(image, label, (x1, y1), color, cfg.font_scale)

    return image


def _build_label(det: Detection, cfg: VisualizeConfig) -> str:
    """Kutu etiketini parcalardan kurar: '#12 araba 0.87'."""
    parts: list[str] = []
    if det.track_id is not None:
        parts.append(f"#{det.track_id}")
    if cfg.show_class_name:
        parts.append(det.cls_name)
    if cfg.show_conf:
        parts.append(f"{det.conf:.2f}")
    if det.depth is not None:
        # Birim YOK. Bu deger goreli - tek kameradan mutlak mesafe olculemez.
        # "m" yazmak kolay olurdu ama olmayan bir kesinlik iddia ederdi;
        # olcek kalibrasyonu Hafta 5'in isi.
        parts.append(f"~{det.depth:.1f}")
    return " ".join(parts)


def _draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
) -> None:
    """Kutunun sol ust kosesine dolu zeminli etiket yazar."""
    x, y = anchor
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = 3

    # Kutu goruntunun ust kenarina yapisiksa etiketi kutunun icine al.
    top = y - text_h - baseline - pad
    if top < 0:
        top = y
    bottom = top + text_h + baseline + pad

    cv2.rectangle(image, (x, top), (x + text_w + 2 * pad, bottom), color, -1)
    cv2.putText(
        image,
        text,
        (x + pad, bottom - baseline - 1),
        _FONT,
        font_scale,
        _text_color_for(color),
        thickness,
        cv2.LINE_AA,
    )


def _text_color_for(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Zemin rengine gore okunabilir yazi rengi (siyah ya da beyaz)."""
    b, g, r = bg
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_trails(
    image: np.ndarray,
    trails: dict[int, list[tuple[int, float, float]]],
    classes: dict[int, int] | None = None,
    *,
    max_thickness: int = 3,
    max_gap: int = 2,
) -> np.ndarray:
    """Takip edilen nesnelerin gecmis zemin temas noktalarini cizgi olarak cizer.

    Iz, eskiden yeniye dogru kalinlasir; boylece hareket yonu tek bakista
    okunur ve durgun nesnelerin izi nokta gibi kalir.

    Ardisik iki nokta arasinda `max_gap`'ten fazla kare varsa cizgi kesilir.
    Bu onemli: bir iz okluzyon yuzunden kaybolup baska bir noktada geri
    geldiginde, aradaki bosluk duz cizgiyle doldurulursa nesnenin hic
    gitmedigi bir yol cizilmis olur - hata analizinde bu, gercek bir kimlik
    atlamasiyla karistirilir.

    Args:
        image: Uzerine cizilecek kare.
        trails: iz kimligi -> (kare_no, x, y) listesi, eskiden yeniye sirali.
        classes: iz kimligi -> sinif id'si. Verilirse iz nesnenin sinif
            renginde cizilir; verilmezse notr gri kullanilir.
        max_thickness: En yeni segmentin kalinligi.
        max_gap: Bu kadar kareye kadar olan boslukta cizgi surdurulur.
    """
    for track_id, points in trails.items():
        if len(points) < 2:
            continue

        color = class_color(classes[track_id]) if classes and track_id in classes else _DEFAULT_COLOR
        span = len(points) - 1

        for i in range(span):
            frame_a, xa, ya = points[i]
            frame_b, xb, yb = points[i + 1]
            if frame_b - frame_a > max_gap:
                continue

            # Segment ne kadar yeniyse o kadar kalin.
            thickness = max(1, int(round(max_thickness * (i + 1) / span)))
            cv2.line(
                image,
                (int(round(xa)), int(round(ya))),
                (int(round(xb)), int(round(yb))),
                color,
                thickness,
                cv2.LINE_AA,
            )

    return image


def colorize_depth(disparity: np.ndarray, colormap: str = "INFERNO") -> np.ndarray:
    """[0,1] araligindaki ters derinlik haritasini renkli gorsele cevirir.

    Girdi disparity oldugu icin acik renkler YAKIN, koyu renkler UZAK demek.
    Bu, "acik = uzak" beklentisinin tersi olabilir; panel basligi bu yuzden
    yonu acikca yaziyor.
    """
    code = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap((np.clip(disparity, 0.0, 1.0) * 255).astype(np.uint8), code)


def stack_panels(
    main: np.ndarray,
    side: np.ndarray,
    *,
    labels: tuple[str, str] | None = None,
) -> np.ndarray:
    """Iki kareyi yan yana birlestirir; sag panel solun yuksekligine olceklenir.

    Hafta 4'te kusbakisi harita da bu fonksiyonla eklenecek - projenin gorsel
    imzasi olan cift panelli cikti burada olusuyor.
    """
    h = main.shape[0]
    if side.shape[0] != h:
        scale = h / side.shape[0]
        side = cv2.resize(side, (max(1, int(round(side.shape[1] * scale))), h), interpolation=cv2.INTER_AREA)

    canvas = np.hstack([main, side])

    if labels:
        for text, x_offset in zip(labels, (0, main.shape[1])):
            draw_hud(canvas, [text], origin=(x_offset + 12, 12), font_scale=0.5)

    return canvas


def draw_hud(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (12, 12),
    font_scale: float = 0.55,
    alpha: float = 0.55,
) -> np.ndarray:
    """Sol ust koseye yari saydam bilgi paneli cizer (FPS, kare no, nesne sayisi)."""
    if not lines:
        return image

    thickness = 1
    sizes = [cv2.getTextSize(line, _FONT, font_scale, thickness)[0] for line in lines]
    line_h = max(h for _, h in sizes) + 8
    panel_w = max(w for w, _ in sizes) + 20
    panel_h = line_h * len(lines) + 10

    x, y = origin
    # Panel goruntu disina tasmasin.
    panel_w = min(panel_w, image.shape[1] - x - 1)
    panel_h = min(panel_h, image.shape[0] - y - 1)
    if panel_w <= 0 or panel_h <= 0:
        return image

    roi = image[y : y + panel_h, x : x + panel_w]
    overlay = np.zeros_like(roi)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x + 10, y + line_h * (i + 1) - 4),
            _FONT,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image
