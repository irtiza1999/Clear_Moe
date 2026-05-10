#!/usr/bin/env python3
"""Run ablation runner and save output to file."""

import subprocess
import sys
import os
from pathlib import Path

os.chdir(r"e:\Research Projects\706")

log_file = Path(r"e:\Research Projects\706\ablation_execution.log")

cmd = [
    sys.executable,
    "scripts/14_ablation_runner.py",
    "--config", "configs/deit_s_imagenet.yaml",
    "--ablation_config", "configs/ablation_deit_s.yaml"
]

print(f"Running ablation runner...")
print(f"Log file: {log_file}")

with open(log_file, "w") as f:
    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)

print(f"Process completed with return code: {result.returncode}")

# Print last lines of log
with open(log_file, "r") as f:
    lines = f.readlines()
    
print("\nLast 100 lines of output:")
print("".join(lines[-100:]))
