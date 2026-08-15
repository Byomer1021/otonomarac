"""Kucuk yardimcilar: cihaz secimi ve asama bazli sure olcumu."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


def resolve_device(device: str = "auto") -> str:
    """"auto" degerini mevcut donanima gore "cuda" veya "cpu"ya cevirir."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> str:
    """Loglarda gostermek icin insan okunur cihaz aciklamasi."""
    if not device.startswith("cuda"):
        return "CPU"
    try:
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        return f"{torch.cuda.get_device_name(index)} (CUDA)"
    except Exception:
        return device


class Profiler:
    """Asama bazli sure toplayici.

    Hafta 8'deki performans tablosu bu siniftan uretilecek; bu yuzden
    olcum en bastan pipeline'in icine gomulu geliyor.
    """

    def __init__(self) -> None:
        self._total: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._total[name] += time.perf_counter() - start
            self._count[name] += 1

    def mean_ms(self, name: str) -> float:
        """Asamanin cagri basina ortalama suresi (ms)."""
        count = self._count[name]
        return (self._total[name] / count) * 1000.0 if count else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Asama adi -> {calls, total_s, mean_ms, fps}."""
        return {
            name: {
                "calls": self._count[name],
                "total_s": total,
                "mean_ms": self.mean_ms(name),
                "fps": self._count[name] / total if total > 0 else 0.0,
            }
            for name, total in self._total.items()
        }

    def format_table(self) -> str:
        """Terminale basmak icin hizalanmis ozet tablosu."""
        rows = self.summary()
        if not rows:
            return "(olcum yok)"

        name_width = max(len("Asama"), max(len(n) for n in rows))
        lines = [f"{'Asama'.ljust(name_width)} | {'Cagri':>6} | {'Ort (ms)':>9} | {'FPS':>7}"]
        lines.append("-" * len(lines[0]))
        for name, stats in sorted(rows.items(), key=lambda kv: -kv[1]["total_s"]):
            lines.append(
                f"{name.ljust(name_width)} | {int(stats['calls']):>6} | "
                f"{stats['mean_ms']:>9.1f} | {stats['fps']:>7.1f}"
            )
        return "\n".join(lines)
