import sys
import time
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) > 1:
        out_csv = Path(sys.argv[1])
    else:
        out_csv = Path("results/telemetry/vram_test2_sealed.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
            except Exception:
                pass
            time.sleep(1.0)

if __name__ == "__main__":
    main()

