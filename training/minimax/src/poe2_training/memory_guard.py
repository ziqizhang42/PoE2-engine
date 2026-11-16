"""Estimate a training or evaluation command's peak memory and refuse work that cannot fit."""

from __future__ import annotations

import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

GIB = 1 << 30
# Set to "off" to skip the estimate check; the workflow's hard memory limit still applies.
ESTIMATE_VARIABLE = "POE2_MEMORY_ESTIMATE"
# A size such as "20G" or "20480M" overrides the hard limit on each workflow command, which is
# otherwise physical memory less a reserve for the rest of the system; "off" runs commands
# without one.
LIMIT_VARIABLE = "POE2_TRAINING_MEMORY_LIMIT"
# Left to the operating system and the session: a command at the limit must not make them swap.
SYSTEM_RESERVE = 4 * GIB


class MemoryGuardError(ValueError):
    """Raised when the estimated peak memory does not fit what the machine has free."""


@dataclass(frozen=True)
class MemoryProfile:
    """Peak bytes of one command: a fixed base plus bytes per record of each kind."""

    host_base: int
    host_train: int
    host_validation: int
    host_piece: int = 0
    device_base: int = 0
    device_train: int = 0
    device_validation: int = 0
    device_piece: int = 0


# Fitted to the peaks measured on the v0.2 corpus and tiled copies of it, from 189k to 4.2M
# train records with 74,157 validation records (build/work-snapshots/step-6-bench and
# step-7-bench), rounded up where a fit underestimated a measurement. Dense holds the float32
# counts and, while it standardizes, a float64 copy: 541 + 6,864 + 13,728 bytes per train
# record, exactly the measured slope. Lookup's host slope is the larger of its GPU and CPU
# fits; pieces scale with the piece as well as the records.
TRAINING_PROFILES = {
    "dense": MemoryProfile(
        host_base=int(1.19 * GIB), host_train=21_133, host_validation=7_405,
        device_base=int(0.07 * GIB), device_train=7_040, device_validation=6_950),
    "pieces": MemoryProfile(
        host_base=int(2.10 * GIB), host_train=480, host_validation=1_000, host_piece=2_000,
        device_base=int(0.07 * GIB), device_train=200, device_validation=250,
        device_piece=21_000),
    "lookup": MemoryProfile(
        host_base=int(1.79 * GIB), host_train=3_320, host_validation=1_000,
        device_base=int(0.10 * GIB), device_train=1_170, device_validation=250),
}
# Sealed evaluation builds float32 counts for the test split and converts them to float64.
EVALUATION_PROFILE = MemoryProfile(host_base=int(1.1 * GIB), host_train=21_133,
                                   host_validation=0)


@dataclass(frozen=True)
class MemoryEstimate:
    host: int
    device: int


def estimate(profile: MemoryProfile, records: int, validation_records: int = 0,
             piece_records: int = 0) -> MemoryEstimate:
    return MemoryEstimate(
        host=(profile.host_base + profile.host_train * records +
              profile.host_validation * validation_records + profile.host_piece * piece_records),
        device=(profile.device_base + profile.device_train * records +
                profile.device_validation * validation_records +
                profile.device_piece * piece_records))


def training_estimate(method: str, records: int, validation_records: int,
                      piece_records: int | None = None) -> MemoryEstimate:
    """Return the estimated peak host and device bytes of one pattern training."""
    if method == "dense" and piece_records is not None:
        return estimate(TRAINING_PROFILES["pieces"], records, validation_records, piece_records)
    return estimate(TRAINING_PROFILES[method], records, validation_records)


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        values[name] = int(rest.split()[0]) * 1024
    return values


def _cgroup_headroom() -> int | None:
    """Return the room left under the tightest memory.max of this process's cgroups."""
    try:
        line = next(item for item in Path("/proc/self/cgroup").read_text().splitlines()
                    if item.startswith("0::"))
    except (OSError, StopIteration):
        return None
    directory = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    headroom: int | None = None
    while True:
        try:
            limit = (directory / "memory.max").read_text().strip()
            current = int((directory / "memory.current").read_text())
        except (OSError, ValueError):
            limit = "max"
            current = 0
        if limit != "max":
            room = int(limit) - current
            headroom = room if headroom is None else min(headroom, room)
        if directory == Path("/sys/fs/cgroup") or directory.parent == directory:
            return headroom
        directory = directory.parent


def _resident() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    cgroup_headroom: int | None
    resident: int

    @property
    def free(self) -> int:
        return (self.available if self.cgroup_headroom is None else
                min(self.available, self.cgroup_headroom))

    @property
    def margin(self) -> int:
        return max(GIB, self.total // 20)


def host_memory() -> HostMemory:
    info = _meminfo()
    return HostMemory(total=info["MemTotal"], available=info["MemAvailable"],
                      cgroup_headroom=_cgroup_headroom(), resident=_resident())


def device_memory(device: Any) -> tuple[int, int] | None:
    """Return free and total bytes of a GPU device, or None on the CPU."""
    if getattr(device, "type", "cpu") != "cuda":
        return None
    import torch

    free, total = torch.cuda.mem_get_info(device)
    return int(free), int(total)


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def check_memory(work: str, needed: MemoryEstimate, *, device: Any = None,
                 remedies: str = "", host: HostMemory | None = None,
                 gpu: tuple[int, int] | None = None) -> None:
    """Raise MemoryGuardError if the estimate does not fit, before anything large exists."""
    if os.environ.get(ESTIMATE_VARIABLE, "").lower() == "off":
        return
    host = host or host_memory()
    growth = max(0, needed.host - host.resident)
    usable = host.free - host.margin
    problems = []
    if growth > usable:
        limit = ("" if host.cgroup_headroom is None or host.cgroup_headroom >= host.available
                 else " within this command's memory limit")
        problems.append(
            f"about {_gib(needed.host)} of host memory, but {_gib(max(0, usable))} is free"
            f"{limit} after a {_gib(host.margin)} margin")
    gpu = gpu if gpu is not None else device_memory(device)
    if gpu is not None:
        free, total = gpu
        margin = max(GIB // 4, total // 20)
        if needed.device > free - margin:
            problems.append(
                f"about {_gib(needed.device)} of GPU memory, but {_gib(max(0, free - margin))} "
                f"is free after a {_gib(margin)} margin")
    if problems:
        raise MemoryGuardError(
            f"{work} needs {' and '.join(problems)}; nothing was allocated."
            f"{' ' + remedies if remedies else ''}")


def parse_size(text: str) -> int:
    """Return the bytes in a size such as "20G", "512M", or "1048576"."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?\s*", text, re.IGNORECASE)
    if match is None:
        raise MemoryGuardError(f"not a memory size: {text!r}; use for example 20G")
    power = " KMGT".index(match.group(2).upper() or " ")
    return int(float(match.group(1)) * (1 << (10 * power)))


def command_memory_limit(environment: Mapping[str, str] = os.environ) -> int | None:
    """Return the host memory limit for each workflow command, or None when switched off."""
    value = environment.get(LIMIT_VARIABLE, "").strip()
    if value.lower() == "off":
        return None
    if value:
        return parse_size(value)
    total = _meminfo()["MemTotal"]
    return max(total - SYSTEM_RESERVE, total // 2)


@dataclass(frozen=True)
class ScopeLimit:
    """Runs commands in systemd user scopes under a host memory limit and without swap."""

    limit: int

    def command(self, unit: str, command: list[str]) -> list[str]:
        # systemd-run moves itself into the scope and then executes the command in place, so
        # the process, its group, its descriptors, and its exit status are the command's own.
        return ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
                "-p", f"MemoryMax={self.limit}", "-p", "MemorySwapMax=0", "--", *command]

    def exceeded(self, unit: str) -> bool:
        """Return whether the kernel killed the scope's command for exceeding the limit."""
        try:
            result = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=Result", "--value"],
                capture_output=True, text=True, timeout=30, check=False)
            subprocess.run(["systemctl", "--user", "reset-failed", unit],
                           capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip() == "oom-kill"


def scope_unit(prefix: str = "poe2") -> str:
    return f"{prefix}-{os.getpid()}-{secrets.token_hex(6)}.scope"


def probe_scope_limit(limit: int) -> str | None:
    """Return why scoped memory limits do not work here, or None once a probe sees them."""
    script = ('p=/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup); '
              'cat "$p/memory.max" "$p/memory.swap.max"')
    try:
        result = subprocess.run(ScopeLimit(limit).command(scope_unit("poe2-probe"),
                                                          ["sh", "-c", script]),
                                capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"systemd-run is not usable: {error}"
    if result.returncode != 0:
        return f"systemd-run failed: {(result.stderr or result.stdout).strip()[:300]}"
    values = result.stdout.split()
    if (len(values) != 2 or not values[0].isdigit() or values[1] != "0" or
            not limit - (1 << 20) < int(values[0]) <= limit):
        return f"the scope did not apply the limits (memory.max and memory.swap.max: {values})"
    return None
