#!/usr/bin/env python3
"""Simple wrapper to run ablation runner and capture output."""

import subprocess
import sys
import os

os.chdir(r"e:\Research Projects\706")

cmd = [
    sys.executable,
    "scripts/14_ablation_runner.py",
    "--config", "configs/deit_s_imagenet.yaml",
    "--ablation_config", "configs/ablation_deit_s.yaml"
]

print(f"Running: {' '.join(cmd)}")
print("=" * 80)

result = subprocess.run(cmd, timeout=3600)
sys.exit(result.returncode)
