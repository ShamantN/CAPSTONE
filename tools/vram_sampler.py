"""
MMCCP 1 Hz VRAM Telemetry Sampler
Continuously queries nvidia-smi every 1.0s and logs timestamp, memory.used (MB), memory.total (MB), utilization.gpu (%)
Outputs to results/telemetry/vram_<date>.csv
"""
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

def main():
    out_dir = Path("results/telemetry")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"vram_{date_str}.csv"
    
    print(f"[VRAM Sampler] Writing to {out_csv}", flush=True)
    
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("timestamp,memory_used_mb,memory_total_mb,gpu_util_percent\n")
        f.flush()
        
        while True:
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if res.returncode == 0:
                    line = res.stdout.strip()
                    if line:
                        f.write(line + "\n")
                        f.flush()
            except Exception as e:
                f.write(f"# Error: {e}\n")
                f.flush()
            time.sleep(1.0)

if __name__ == "__main__":
    main()
