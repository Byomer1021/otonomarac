"""Nesne takip katmani (Hafta 2).

Ultralytics'in `model.track()` kisayolu yerine `BYTETracker` dogrudan
kullaniliyor. Sebep: kisayol tespit ve takibi tek cagriya gomuyor, boylece
(a) iki asamanin suresi ayri olculemiyor, (b) tespit katmaninin ciktisi
takibe girmeden once elden gecirilemiyor. Mimari diyagramda bu iki kutu
ayri cizildigi icin kodda da ayri duruyorlar.

Bedeli: Ultralytics'in ic API'sine bagimlilik. Bu bagimlilik tek dosyada
(asagidaki `_ResultsAdapter`) toplandi; kutuphane arayuzu degisirse
duzeltilecek tek yer orasi.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .config import TrackingConfig
from .detection import Detection

#: BYTETracker.update() ciktisindaki sutun sirasi.
#: [x1, y1, x2, y2, track_id, score, cls, detection_index]
_COL_TRACK_ID = 4
_COL_DET_INDEX = 7


class _ResultsAdapter:
    """`Detection` listesini BYTETracker'in bekledigi arayuze cevirir.

    BYTETracker `results.xywh`, `results.conf`, `results.cls`, `len(results)`
    ve boolean maske ile indeksleme bekler - Ultralytics'in `Boxes` sinifinin
    kucuk bir alt kumesi.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> _ResultsAdapter:
        if not detections:
            empty = np.zeros((0, 4), dtype=np.float32)
            return cls(empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))

        xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # merkez x
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # merkez y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]        # genislik
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]        # yukseklik

        return cls(
            xywh,
            np.array([d.conf for d in detections], dtype=np.float32),
            np.array([d.cls_id for d in detections], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> _ResultsAdapter:
        return _ResultsAdapter(self.xywh[mask], self.conf[mask], self.cls[mask])


@dataclass
class TrackStats:
    """Kimlik surekliligi olcumleri - Hafta 2'nin asil ciktisi.

    Takibin "calisiyor gorunmesi" yetmez; kac kimlik uretildigi ve bunlarin
    ne kadar yasadigi olculmeden TTC hesabina guvenilemez.
    """

    #: iz kimligi -> o izin gorulduğu kare sayisi
    frames_seen: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    #: iz kimligi -> (ilk kare, son kare)
    first_frame: dict[int, int] = field(default_factory=dict)
    last_frame: dict[int, int] = field(default_factory=dict)
    #: Kare basina es zamanli iz sayisi
    concurrent: list[int] = field(default_factory=list)

    def update(self, frame_index: int, track_ids: list[int]) -> None:
        for tid in track_ids:
            self.frames_seen[tid] += 1
            self.first_frame.setdefault(tid, frame_index)
            self.last_frame[tid] = frame_index
        self.concurrent.append(len(track_ids))

    def fragmented(self, min_track_len: int) -> list[int]:
        """Cok kisa yasamis izler - kimlik atlamasinin en gorunur belirtisi."""
        return [tid for tid, n in self.frames_seen.items() if n < min_track_len]

    def gapped(self) -> list[int]:
        """Ilk ve son kare arasi, gorulduğu kare sayisindan fazla olan izler.

        Bu fark izin arada kaybolup geri geldigini gosterir (okluzyon ya da
        tespit kaybi). Kimlik korunmus ama iz sureklilik acisindan delikli.
        """
        return [
            tid
            for tid in self.frames_seen
            if (self.last_frame[tid] - self.first_frame[tid] + 1) > self.frames_seen[tid]
        ]

    def format_report(self, min_track_len: int) -> str:
        if not self.frames_seen:
            return "Takip: hic iz olusmadi."

        lengths = sorted(self.frames_seen.values())
        fragmented = self.fragmented(min_track_len)
        gapped = self.gapped()

        lines = [
            f"Toplam benzersiz kimlik    : {len(self.frames_seen)}",
            f"Es zamanli iz (ort / maks) : {statistics.mean(self.concurrent):.1f} / {max(self.concurrent)}",
            f"Iz uzunlugu (medyan / maks): {statistics.median(lengths):.0f} / {max(lengths)} kare",
            f"Parcalanmis iz (<{min_track_len} kare)  : {len(fragmented)}"
            f"  ({len(fragmented) / len(self.frames_seen) * 100:.0f}%)",
            f"Delikli iz (arada kaybolan): {len(gapped)}"
            f"  ({len(gapped) / len(self.frames_seen) * 100:.0f}%)",
        ]

        # Yorum satiri: rakamin ne anlama geldigini okuyucuya birakma.
        ratio = len(fragmented) / len(self.frames_seen)
        if ratio > 0.5:
            lines.append(
                "  -> Izlerin yarisindan fazlasi kisa omurlu. TTC hesabi bu haliyle guvenilmez; "
                "new_track_thresh yukseltilmeli veya detektor guveni gozden gecirilmeli."
            )
        elif ratio > 0.25:
            lines.append("  -> Parcalanma belirgin ama TTC icin minimum iz uzunlugu sarti ile tolere edilebilir.")
        else:
            lines.append("  -> Kimlik surekliligi TTC hesabi icin yeterli.")

        return "\n".join(lines)


class ObjectTracker:
    """ByteTrack sarmalayicisi + hareket izi gecmisi."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:  # pragma: no cover - kurulum hatasi
            raise ImportError(
                "ultralytics.trackers bulunamadi. 'pip install -r requirements.txt' calistirin."
            ) from exc

        # BYTETracker bir Namespace bekliyor; bytetrack.yaml'daki alanlarin aynisi.
        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=self.config.track_high_thresh,
                track_low_thresh=self.config.track_low_thresh,
                new_track_thresh=self.config.new_track_thresh,
                track_buffer=self.config.track_buffer,
                match_thresh=self.config.match_thresh,
                fuse_score=self.config.fuse_score,
            )
        )

        self.stats = TrackStats()
        #: Iz gecmisi kare cinsinden tutulur; kaynak videonun FPS'i
        #: bilinene kadar 30 fps varsayilir (bkz. configure_for_fps).
        self._trail_maxlen = self._maxlen_for_fps(30.0)
        #: iz kimligi -> son N (kare_no, x, y) zemin temas noktasi.
        #: Kare numarasi da saklaniyor cunku bir iz kaybolup geri geldiginde
        #: eski nokta ile yeni nokta duz cizgiyle birlestirilmemeli - o cizgi
        #: nesnenin gitmedigi bir yolu gostermis olur.
        self.trails: dict[int, deque[tuple[int, float, float]]] = {}

    def _maxlen_for_fps(self, fps: float) -> int:
        """Iz suresini (saniye) o kaynagin kare sayisina cevirir."""
        if self.config.trail_seconds <= 0 or fps <= 0:
            return 0
        return max(2, int(round(self.config.trail_seconds * fps)))

    def configure_for_fps(self, fps: float) -> None:
        """Kaynak videonun kare hizi ogrenildiginde cagrilir.

        Ilk iz olusmadan once cagrilmali - deque'lerin maxlen'i sonradan
        degistirilemez, ama izler tembel olusturuldugu icin dongu baslamadan
        ayarlamak yeterli.
        """
        self._trail_maxlen = self._maxlen_for_fps(fps)

    def update(self, detections: list[Detection], frame_index: int) -> list[Detection]:
        """Tespitlere kimlik atar.

        Kimlik alamayan tespitler dusurulur - takip acikken pipeline'in
        para birimi "izlenen nesne"dir, ciplak tespit degil.
        """
        rows = self._tracker.update(_ResultsAdapter.from_detections(detections))

        tracked: list[Detection] = []
        for row in np.atleast_2d(rows) if len(rows) else []:
            det_index = int(row[_COL_DET_INDEX])
            if not 0 <= det_index < len(detections):
                # Savunmaci: Ultralytics indeks sozlesmesini degistirirse
                # sessizce yanlis nesneye kimlik yazmaktansa atla.
                continue
            # Kutu geometrisi tespit katmanindan geldigi gibi kalir; takip
            # katmani sadece kimlik atar. Kalman ile duzeltilmis kutuyu
            # kullanmak gorsel olarak daha yumusak olurdu ama tespit
            # ciktisini gizler - hata analizinde ayrimi korumak daha degerli.
            detection = detections[det_index]
            detection.track_id = int(row[_COL_TRACK_ID])
            tracked.append(detection)

        self._update_trails(tracked, frame_index)
        self.stats.update(frame_index, [d.track_id for d in tracked if d.track_id is not None])
        return tracked

    def _update_trails(self, tracked: list[Detection], frame_index: int) -> None:
        if self._trail_maxlen <= 0:
            return

        for det in tracked:
            if det.track_id is None:
                continue
            trail = self.trails.get(det.track_id)
            if trail is None:
                trail = deque(maxlen=self._trail_maxlen)
                self.trails[det.track_id] = trail
            x, y = det.bottom_center
            trail.append((frame_index, x, y))

        # Kaybolan izlerin gecmisi silinmiyor: iz geri gelirse kaldigi yerden
        # devam etsin. Bellek sinirli, cunku her deque maxlen'e bagli.

    def trail_for(self, track_id: int) -> list[tuple[int, float, float]]:
        return list(self.trails.get(track_id, ()))

    def active_trails(self, tracked: list[Detection]) -> dict[int, list[tuple[int, float, float]]]:
        """Sadece bu karede gorulen izlerin gecmisi - cizim katmani icin."""
        return {
            det.track_id: list(self.trails[det.track_id])
            for det in tracked
            if det.track_id is not None and det.track_id in self.trails
        }
