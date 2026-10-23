"""Spend cap for the public Modal demo (demo/modal_app.py). Standard library only, unit-tested on the Mac.

Modal bills a container for its whole uptime: from boot to exit, the idle scaledown window included,
at the higher of what it requested and what it used. So the ledger counts container uptime, not
request time. Every container (GPU scorer, web page, daily keepalive) writes its own running total
under its own key, so writers never overwrite each other:

    "<YYYY-MM>|<kind>|<container id>" -> {"usd": float, "start": epoch s, "t": epoch s of the last write}

The month's spend is the sum of that month's keys. A GPU container computes its own cost exactly in
process; a key is at most HEARTBEAT_S stale for any other reader, which the admission checks cover
with MARGIN_S.

Prices: `modal billing rates` read 2026-10-19 (the same numbers as modal.com/pricing that day, per
second there: L4 $0.000222, CPU $0.0000131 per physical core, memory $0.00000222 per GiB). Modal
bills at least 0.125 core and 128 MiB per container (modal.com/pricing FAQ, read 2026-10-19).
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, MutableMapping

PRICES_READ = "2026-10-19"
GPU_USD_PER_H = {"L4": 0.80, "T4": 0.59, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "H100": 3.95}
CPU_USD_PER_CORE_H = 0.0473
MEM_USD_PER_GIB_H = 0.008
MIN_CORES, MIN_GIB = 0.125, 0.125

HEARTBEAT_S = 15  # how often a container writes its own total
MARGIN_S = 45     # covers a stale heartbeat plus the gap between Modal's boot and our first timestamp


def rate_per_s(gpu: str | None, cores: float, mem_gib: float) -> float:
    """USD per second of uptime for one container. Pass the container's limits (not requests): Modal bills
    the higher of request and use, and use never exceeds the limit, so this is an upper bound."""
    gpu_h = GPU_USD_PER_H[gpu] if gpu else 0.0
    return (gpu_h + max(cores, MIN_CORES) * CPU_USD_PER_CORE_H + max(mem_gib, MIN_GIB) * MEM_USD_PER_GIB_H) / 3600


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def cap_for(month: str, caps: dict[str, float], default: float) -> float:
    """The cap in USD for a calendar month (UTC): an explicit entry, else the default."""
    return float(caps.get(month, default))


class Ledger:
    """Container uptime in USD, on any str -> dict mapping (a modal.Dict in the app, a dict in tests)."""

    def __init__(self, store: MutableMapping[str, Any]) -> None:
        self.store = store

    @staticmethod
    def key(month: str, kind: str, cid: str) -> str:
        return f"{month}|{kind}|{cid}"

    def record(self, kind: str, cid: str, start: float, now: float, rate: float, boot_s: float) -> float:
        """Write this container's total so far: (now - start + boot_s) * rate, under the month it started in."""
        usd = round((now - start + boot_s) * rate, 6)
        self.store[self.key(month_of(start), kind, cid)] = {"usd": usd, "start": start, "t": now}
        return usd

    def entries(self, month: str) -> dict[str, dict]:
        out = {}
        for k, v in self.store.items():
            if isinstance(k, str) and k.startswith(month + "|") and isinstance(v, dict):
                out[k] = v
        return out

    def spent(self, month: str, exclude: str | None = None) -> float:
        return round(sum(float(v.get("usd", 0.0)) for k, v in self.entries(month).items() if k != exclude), 6)

    def by_kind(self, month: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.entries(month).items():
            kind = k.split("|")[1]
            out[kind] = round(out.get(kind, 0.0) + float(v.get("usd", 0.0)), 6)
        return out

    def refresh(self, months: list[str]) -> int:
        """Re-write every entry of these months (a Dict entry expires after 7 days without a read or write)."""
        n = 0
        for m in months:
            for k, v in self.entries(m).items():
                self.store[k] = v
                n += 1
        return n


def gpu_reserve_usd(rate: float, request_s: float, scaledown_s: float) -> float:
    """What one more request can add on a warm GPU container: the request itself (bounded by its timeout),
    the idle scaledown window that follows it, and the margin."""
    return (request_s + scaledown_s + MARGIN_S) * rate


def admit_gpu(*, cap: float, spent_others: float, own_usd: float, rate: float, request_s: float,
              scaledown_s: float) -> bool:
    """Authoritative check, run inside the single GPU container at the start of each (serialized) call:
    everything already counted, plus this container's exact cost so far, plus the most this call can add."""
    return spent_others + own_usd + gpu_reserve_usd(rate, request_s, scaledown_s) <= cap


def admit_web(*, cap: float, spent: float, warm: bool, rate: float, cold_s: float, request_s: float,
              scaledown_s: float) -> bool:
    """Fast check in the web container before it asks the GPU for anything. A cold GPU also costs its boot
    and model load (cold_s), so a cold start is admitted only when that fits too."""
    extra = 0.0 if warm else (cold_s + MARGIN_S) * rate
    return spent + extra + gpu_reserve_usd(rate, request_s, scaledown_s) <= cap


class RateLimiter:
    """At most `limit` events per `window_s` seconds per client, in memory only. Clients are kept as a short
    salted hash of their IP, never the IP itself."""

    def __init__(self, limit: int, window_s: float, salt: bytes) -> None:
        self.limit, self.window_s, self.salt = limit, window_s, salt
        self.events: dict[str, deque] = {}
        self.lock = threading.Lock()

    def _id(self, client: str) -> str:
        return hashlib.blake2b(client.encode(), key=self.salt[:64], digest_size=8).hexdigest()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        cid = self._id(client)
        with self.lock:
            q = self.events.setdefault(cid, deque())
            while q and now - q[0] > self.window_s:
                q.popleft()
            for stale in [k for k, v in self.events.items() if not v and k != cid]:
                del self.events[stale]
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


class Gate:
    """A bounded queue: at most `active` calls run and `waiting` more wait; anything beyond is refused."""

    def __init__(self, active: int, waiting: int) -> None:
        self.sem = threading.Semaphore(active)
        self.lock = threading.Lock()
        self.inside, self.limit = 0, active + waiting

    def enter(self, timeout: float) -> bool:
        """True once the caller may run; False when the queue is full or the wait passed `timeout` seconds."""
        with self.lock:
            if self.inside >= self.limit:
                return False
            self.inside += 1
        if self.sem.acquire(timeout=timeout):
            return True
        with self.lock:
            self.inside -= 1
        return False

    def leave(self) -> None:
        self.sem.release()
        with self.lock:
            self.inside -= 1
