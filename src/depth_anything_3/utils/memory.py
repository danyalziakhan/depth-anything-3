"""
GPU memory utility helpers.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Return a snapshot of current GPU memory usage or None if CUDA not available.

    Keys in returned dict: total_gb, allocated_gb, reserved_gb, free_gb, utilization
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory

        return {
            "total_gb": total_memory / 1024**3,
            "allocated_gb": allocated_memory / 1024**3,
            "reserved_gb": reserved_memory / 1024**3,
            "free_gb": free_memory / 1024**3,
            "utilization": (reserved_memory / total_memory) * 100,
        }
    except Exception:
        return None


def cleanup_cuda_memory() -> None:
    """Perform a robust GPU cleanup sequence.

    This includes synchronizing, emptying caches, collecting IPC handles and
    running the Python garbage collector. Use this instead of a raw
    ``torch.cuda.empty_cache()`` where you need reliable freeing of GPU memory
    between model loads or in error handling paths.
    """
    try:
        if torch.cuda.is_available():
            mem_before = get_gpu_memory_info()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Collect cross-process cuda resources
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # Older PyTorch versions or non-cuda devices may not support
                # ipc_collect (no-op if not available)
                pass
            gc.collect()

            mem_after = get_gpu_memory_info()
            if mem_before and mem_after:
                freed = mem_before["reserved_gb"] - mem_after["reserved_gb"]
                print(
                    f"CUDA cleanup: freed {freed:.2f}GB, "
                    f"available: {mem_after['free_gb']:.2f}GB/{mem_after['total_gb']:.2f}GB"
                )
            else:
                print("CUDA memory cleanup completed")
    except Exception as e:
        print(f"Warning: CUDA cleanup failed: {e}")
