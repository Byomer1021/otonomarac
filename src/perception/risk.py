"""Goreli hiz ve carpismaya kalan sure (Hafta 5).

Her izlenen nesnenin zemin duzlemindeki gecmisinden yaklasma hizi cikarilir ve
mesafe / yaklasma hizi oranindan TTC hesaplanir.

Uc tasarim karari, uctu de Hafta 2 ve 4'te ortaya cikan sorunlardan geliyor:

1. **Zaman kare sayisiyla degil saniyeyle olculur.** Izlerin %29'u delikli -
   nesne okluzyonda kaybolup geri geliyor. Iki gozlem arasinda 1 kare de
   olabilir 8 kare de; farki kare sayarak degil zaman damgasindan almak sart.

2. **Hiz iki noktadan degil, pencereye dogru fit edilerek cikarilir.** Ardisik
   iki kare arasindaki fark, tespit kutusunun titremesi yaninda kaybolur.

3. **Yetersiz veride sayi uretilmez.** Kisa iz, dusuk yaklasma hizi ya da cok
   uzak nesne icin TTC None doner. Bir sayi ekranda gorundugu anda olculmus
   sayilir; uretilmemesi, guvenilmez uretilmesinden iyidir.

TTC ve olcek belirsizligi
-------------------------

Hafta 4, boylamsal olcegin (`bev.quad_depth_m`) goruntuden belirlenemedigini
gosterdi: yanal referanslar - arac genisligi, serit genisligi - odak uzakligi
sadelestigi icin yalnizca kamera yuksekligini verir, mesafeyi vermez.

TTC bu belirsizlikten **etkilenmez.** Butun mesafeler bilinmeyen bir k
katsayisiyla carpilirsa yaklasma hizi da ayni k ile carpilir:

    TTC = mesafe / yaklasma_hizi = (k*d) / (k*v) = d / v

Yani haritadaki metre degerleri bir olcek katsayisi kadar belirsizken,
saniye cinsinden TTC metrik olarak dogrudur. Projenin en cok ise yarayan
ciktisi, en zayif varsayimindan bagimsiz cikti.

Istisna: `min_distance_m` esigi boylamsal olcekte tanimli ve o belirsizlikten
etkilenir. `path_half_width_m` etkilenmez - yanal olcek kalibre edildi.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import RiskConfig
from .detection import Detection


class RiskLevel(str, Enum):
    """Nesnenin risk sinifi."""

    NONE = "yok"
    WARNING = "uyari"
    CRITICAL = "kritik"


@dataclass
class Motion:
    """Bir izin cikarilmis hareket durumu."""

    track_id: int
    #: Ego'ya gore boylamsal mesafe (metre, referans satirindan)
    distance_m: float
    #: Yaklasma hizi (m/s, POZITIF = yaklasiyor)
    closing_speed: float | None
    #: Carpismaya kalan sure (saniye). Hesaplanamadiysa None.
    ttc_s: float | None
    level: RiskLevel
    #: Hiz tahmininde kullanilan gozlem sayisi
    observations: int


class RiskEstimator:
    """Iz gecmisinden hiz ve TTC cikarir."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        #: iz kimligi -> (zaman_s, mesafe_m) gecmisi
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=120)
        )
        self.motions: dict[int, Motion] = {}

        #: Calistirma boyunca biriken sayimlar - `summary()` bunlari raporlar.
        #: Tek karenin durumu ozet degildir; ozetin tum videoyu kapsamasi gerek.
        self._frames = 0
        self._with_speed = 0
        self._with_ttc = 0
        self._tracks_seen: set[int] = set()
        self._critical_tracks: set[int] = set()
        self._warning_tracks: set[int] = set()
        self._min_ttc: tuple[float, int, float, float] | None = None

    def update(
        self, detections: list[Detection], timestamp: float, ego_x: float = 0.0
    ) -> dict[int, Motion]:
        """Bu karedeki nesneler icin hareket durumunu gunceller.

        Args:
            detections: Zemin konumu atanmis nesneler.
            timestamp: Karenin kaynak videodaki zamani (saniye).
            ego_x: Ego aracin haritadaki yanal konumu (metre).
        """
        self.motions = {}

        for det in detections:
            if det.track_id is None or det.bev_xy is None:
                continue

            lateral, distance = det.bev_xy
            history = self._history[det.track_id]
            # Gecmis her nesne icin tutulur - koridor disindaki bir nesne
            # sonra koridora girerse hazir bir gecmisi olsun.
            history.append((timestamp, distance))

            in_path = abs(lateral - ego_x) <= self.config.path_half_width_m
            if (
                not in_path
                or distance < self.config.min_distance_m
                or det.conf < self.config.min_confidence
            ):
                motion = Motion(det.track_id, distance, None, None, RiskLevel.NONE, len(history))
                self.motions[det.track_id] = motion
                det.ttc = None
                det.risk = RiskLevel.NONE.value
                continue

            motion = self._estimate(det.track_id, distance, history, timestamp)
            self.motions[det.track_id] = motion
            det.ttc = motion.ttc_s
            det.risk = motion.level.value
            self._record(motion)

        self._frames += 1
        return self.motions

    def _record(self, motion: Motion) -> None:
        self._tracks_seen.add(motion.track_id)
        if motion.closing_speed is not None:
            self._with_speed += 1
        if motion.ttc_s is None:
            return

        self._with_ttc += 1
        if motion.level is RiskLevel.CRITICAL:
            self._critical_tracks.add(motion.track_id)
        elif motion.level is RiskLevel.WARNING:
            self._warning_tracks.add(motion.track_id)

        if self._min_ttc is None or motion.ttc_s < self._min_ttc[0]:
            self._min_ttc = (
                motion.ttc_s,
                motion.track_id,
                motion.distance_m,
                motion.closing_speed or 0.0,
            )

    def _estimate(
        self,
        track_id: int,
        distance: float,
        history: deque[tuple[float, float]],
        now: float,
    ) -> Motion:
        # Yalnizca son `history_seconds` icindeki gozlemler. Daha eskisi,
        # nesnenin o zamandan beri hizlanmis olabilecegi icin yaniltir.
        window = [(t, d) for t, d in history if now - t <= self.config.history_seconds]

        if len(window) < self.config.min_observations:
            return Motion(track_id, distance, None, None, RiskLevel.NONE, len(window))

        times = np.array([t for t, _ in window])
        dists = np.array([d for _, d in window])
        if times.max() - times.min() < 1e-3:
            return Motion(track_id, distance, None, None, RiskLevel.NONE, len(window))

        # Mesafenin zamana gore egimi. Negatif egim = mesafe azaliyor = yaklasiyor.
        coeffs = np.polyfit(times, dists, 1)
        slope = float(coeffs[0])
        closing = -slope

        # Fit kalitesi kapisi: artik buyukse mesafe duzgun degismiyor demektir
        # ve egimden cikarilan "hiz" gercek hareket degil gurultudur.
        residual = float(np.sqrt(np.mean((dists - np.polyval(coeffs, times)) ** 2)))
        if residual > self.config.max_fit_residual_ratio * max(distance, 1.0):
            return Motion(track_id, distance, None, None, RiskLevel.NONE, len(window))

        if closing < self.config.min_closing_speed:
            # Uzaklasiyor ya da duragan: TTC tanimsiz, risk yok.
            return Motion(track_id, distance, closing, None, RiskLevel.NONE, len(window))

        ttc = distance / closing
        if ttc > self.config.ttc_max_s or ttc < 0:
            return Motion(track_id, distance, closing, None, RiskLevel.NONE, len(window))

        if ttc <= self.config.ttc_critical_s:
            level = RiskLevel.CRITICAL
        elif ttc <= self.config.ttc_warning_s:
            level = RiskLevel.WARNING
        else:
            level = RiskLevel.NONE

        return Motion(track_id, distance, closing, ttc, level, len(window))

    def summary(self) -> str:
        """Tum calistirmayi kapsayan ozet."""
        if not self._tracks_seen:
            return "Risk: zemin konumu olan hicbir iz olusmadi."

        total = self._with_speed + 0
        lines = [
            f"Guzergah koridorundaki iz: {len(self._tracks_seen)}",
            f"Hiz cikarilan gozlem     : {self._with_speed}",
            f"TTC uretilen gozlem      : {self._with_ttc}"
            + (f"  (%{self._with_ttc / total * 100:.0f})" if total else ""),
            f"Uyari veren iz (<{self.config.ttc_warning_s:.0f}s)  : {len(self._warning_tracks)}",
            f"Kritik iz (<{self.config.ttc_critical_s:.0f}s)      : {len(self._critical_tracks)}",
        ]

        if self._min_ttc is not None:
            ttc, track_id, distance, closing = self._min_ttc
            lines.append(
                f"En dusuk TTC             : {ttc:.1f}s  (#{track_id}, "
                f"{distance:.1f}m, {closing:.1f} m/s yaklasiyor)"
            )

        # Rakamin ne anlama geldigini okuyucuya birakma.
        if not self._with_ttc:
            lines.append(
                "  -> Hic TTC uretilmedi. Sahnede yaklasan nesne olmayabilir; "
                "ya da izler hiz cikarimi icin fazla kisa."
            )
        return "\n".join(lines)
