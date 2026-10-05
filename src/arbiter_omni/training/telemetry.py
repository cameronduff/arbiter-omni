"""
Lightweight process and system memory telemetry for training diagnostics.

Reads /proc directly (Linux / WSL2) so there is no extra dependency and no
measurable overhead. All functions degrade gracefully to 0.0 on platforms
without /proc.
"""

from __future__ import annotations

import resource
from typing import Tuple


def get_rss_mb() -> float:
    """Resident set size of the current process in MB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def get_system_memory_mb() -> Tuple[float, float]:
    """Returns (total_mb, available_mb) system memory from /proc/meminfo."""
    total = available = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    available = float(line.split()[1]) / 1024.0
                if total and available:
                    break
    except OSError:
        pass
    return total, available


def memory_summary() -> str:
    """Compact one-line memory summary suitable for appending to a log line."""
    total, available = get_system_memory_mb()
    rss = get_rss_mb()
    if total > 0:
        return f"rss={rss:,.0f}MB sys_avail={available:,.0f}/{total:,.0f}MB"
    return f"rss={rss:,.0f}MB"
