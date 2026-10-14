"""
STAGE 2: DYNAMIC MEMORY AND VRAM MANAGEMENT (RTX 3070 Ti / 32GB RAM)
Provides exception-safe VRAM telemetry, context managers, and memory guards.
"""
import os, sys, gc, time, ctypes, logging
from typing import Dict, Tuple, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger("MMCCP.MemoryUtils")

# Windows GlobalMemoryStatusEx for host RAM tracking
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]

def get_host_ram_mb() -> Tuple[float, float, float]:
    """Returns (total_mb, available_mb, used_mb)."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    total_mb = stat.ullTotalPhys / (1024 ** 2)
    avail_mb = stat.ullAvailPhys / (1024 ** 2)
    used_mb = total_mb - avail_mb
    return round(total_mb, 1), round(avail_mb, 1), round(used_mb, 1)

def cleanup_memory(force: bool = False, threshold_mb: float = 1200.0):
    """
    Cleans up host RAM and GPU VRAM garbage.
    Purges evaluation PyTorch cache while leaving Forge model weights untouched in GPU.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            if force or alloc > threshold_mb:
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        import tf_keras
        tf_keras.backend.clear_session()
    except Exception:
        pass

def get_vram_telemetry() -> Dict[str, float]:
    """Returns detailed GPU VRAM telemetry dictionary."""
    telemetry = {
        "vram_total_mb": 8192.0,
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_peak_allocated_mb": 0.0,
        "host_ram_used_mb": 0.0,
        "host_ram_avail_mb": 0.0
    }
    try:
        import torch
        if torch.cuda.is_available():
            telemetry["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
            telemetry["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
            telemetry["vram_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass

    _, avail_mb, used_mb = get_host_ram_mb()
    telemetry["host_ram_used_mb"] = used_mb
    telemetry["host_ram_avail_mb"] = avail_mb
    return telemetry

@contextmanager
def vram_guard(threshold_mb: float = 7400.0, scene_tag: str = "Scene"):
    """
    Context manager tracking generation VRAM & runtime with automatic leak prevention.
    """
    t0 = time.time()
    initial_vram = get_vram_telemetry()
    cleanup_memory(force=False)
    
    logger.info(f"[{scene_tag}] Starting Generation | Initial VRAM: {initial_vram['vram_allocated_mb']}MB allocated, Host RAM used: {initial_vram['host_ram_used_mb']}MB")
    
    try:
        yield initial_vram
    finally:
        elapsed = time.time() - t0
        final_vram = get_vram_telemetry()
        logger.info(f"[{scene_tag}] Finished in {elapsed:.2f}s | Final VRAM: {final_vram['vram_allocated_mb']}MB allocated (Peak: {final_vram['vram_peak_allocated_mb']}MB)")
        cleanup_memory(force=True)

if __name__ == "__main__":
    t = get_vram_telemetry()
    print("Memory Telemetry Test:", t)
