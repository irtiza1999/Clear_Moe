#!/usr/bin/env python3
"""Start ablation runner in background process."""

import subprocess
import sys
import os
from pathlib import Path
import time

os.chdir(r"e:\Research Projects\706")

log_file = Path(r"e:\Research Projects\706\ablation_execution.log")

cmd = [
    sys.executable,
    "scripts/14_ablation_runner.py",
    "--config", "configs/deit_s_imagenet.yaml",
    "--ablation_config", "configs/ablation_deit_s.yaml"
]

print(f"Starting ablation runner in background...")
print(f"Log file: {log_file}")

with open(log_file, "w") as f:
    # Start process but don't wait for it
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Process ID: {proc.pid}")

# Let it run for a bit and check
time.sleep(5)
print(f"Process is running. Check ablation_execution.log for progress.")
