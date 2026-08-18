"""Verify the PyTorch/ROCm environment and benchmark feature ingestion."""

from __future__ import annotations

import argparse
import math
import platform
import sys
import sysconfig
import time
from importlib import metadata
from pathlib import Path


def _milliseconds(begin: float, end: float) -> float:
    return (end - begin) * 1000.0


def _mebibytes(size: int) -> float:
    return size / (1024.0 * 1024.0)


def _synchronize(torch: object, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()  # type: ignore[attr-defined]


def _environment(torch: object, numpy: object, require_gpu: bool) -> str:
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(f"Python 3.14 is required, found {platform.python_version()}")
    if numpy.__version__ != "2.5.1":
        raise RuntimeError(f"NumPy 2.5.1 is required, found {numpy.__version__}")
    torch_base = torch.__version__.split("+")[0]  # type: ignore[attr-defined]
    if torch_base != "2.12.1":
        raise RuntimeError(f"PyTorch 2.12.1 is required, found {torch.__version__}")
    triton_version = metadata.version("triton-rocm")
    if triton_version != "3.7.1":
        raise RuntimeError(f"triton-rocm 3.7.1 is required, found {triton_version}")
    try:
        overlapping_triton = metadata.version("pytorch-triton-rocm")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "redundant pytorch-triton-rocm distribution is installed alongside "
            f"triton-rocm: {overlapping_triton}"
        )
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if free_threaded:
        raise RuntimeError("the primary environment must use regular, non-free-threaded CPython")
    available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
    if require_gpu and not available:
        raise RuntimeError("ROCm GPU is required but torch.cuda.is_available() is false")
    hip_version = torch.version.hip  # type: ignore[attr-defined]
    if hip_version is None or not hip_version.startswith("7.2"):
        raise RuntimeError(f"a ROCm 7.2 PyTorch build is required, found HIP {hip_version}")
    device = "cuda" if available else "cpu"
    device_name = (torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
                   if available else platform.processor())
    print(
        "training_environment"
        f" python={platform.python_version()}"
        f" free_threaded={str(free_threaded).lower()}"
        f" numpy={numpy.__version__}"
        f" torch={torch.__version__}"
        f" triton_rocm={triton_version}"
        f" hip={torch.version.hip}"
        f" gpu_available={str(available).lower()}"
        f" device={device_name!r}"
    )
    return device


def _tensor_smoke(torch: object, device: str) -> None:
    torch.manual_seed(20260818)  # type: ignore[attr-defined]
    torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    begin = time.perf_counter()
    left = torch.randn(  # type: ignore[attr-defined]
        (512, 512), dtype=torch.float32, device=device, requires_grad=True
    )
    right = torch.randn(  # type: ignore[attr-defined]
        (512, 512), dtype=torch.float32, device=device, requires_grad=True
    )
    loss = (left @ right).square().mean()
    loss.backward()
    _synchronize(torch, device)
    end = time.perf_counter()
    loss_value = float(loss.detach().cpu())
    gradient_norm = float(left.grad.detach().norm().cpu())
    if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
        raise RuntimeError("tensor smoke test produced a non-finite result")
    print(
        "tensor_smoke_valid"
        f" device={device}"
        f" elapsed_ms={_milliseconds(begin, end):.3f}"
        f" loss={loss_value:.9f}"
        f" gradient_norm={gradient_norm:.9f}"
    )


def _dataset_smoke(torch: object, dataset_path: Path, device: str,
                   verify_digest: bool) -> None:
    from .dataset import MappedFeatureDataset, to_torch

    begin = time.perf_counter()
    dataset = MappedFeatureDataset(dataset_path, verify_digest=verify_digest)
    mapped = time.perf_counter()
    try:
        batch = dataset.materialize()
        materialized = time.perf_counter()
        cpu_batch = to_torch(batch)
        tensorized = time.perf_counter()
        device_batch = cpu_batch.to(device)
        _synchronize(torch, device)
        transferred = time.perf_counter()

        residual_ok = bool(torch.equal(
            device_batch.residuals,
            device_batch.teacher_values - device_batch.b_values,
        ))
        if not residual_ok:
            raise RuntimeError("device tensors failed the residual identity")
        split_counts = dataset.artifact.split_counts
        print(
            "feature_ingestion_valid"
            f" records={batch.records}"
            f" train={split_counts[0]}"
            f" validation={split_counts[1]}"
            f" test={split_counts[2]}"
            f" digest={str(verify_digest).lower()}"
            f" authenticate_map_ms={_milliseconds(begin, mapped):.3f}"
            f" materialize_ms={_milliseconds(mapped, materialized):.3f}"
            f" tensorize_ms={_milliseconds(materialized, tensorized):.3f}"
            f" transfer_ms={_milliseconds(tensorized, transferred):.3f}"
            f" host_mib={_mebibytes(batch.nbytes):.3f}"
            f" device_mib={_mebibytes(device_batch.nbytes):.3f}"
        )
    finally:
        dataset.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="completed minimax feature artifact")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail unless a ROCm device is available")
    parser.add_argument("--skip-digest", action="store_true",
                        help="skip the full binary SHA-256 pass for timing diagnostics")
    arguments = parser.parse_args()

    try:
        import numpy
        import torch

        device = _environment(torch, numpy, arguments.require_gpu)
        _tensor_smoke(torch, device)
        _dataset_smoke(torch, arguments.dataset, device, not arguments.skip_digest)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"training smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
