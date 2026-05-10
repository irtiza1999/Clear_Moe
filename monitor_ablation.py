#!/usr/bin/env python3
"""Monitor ablation runner progress and save final status."""

import os
import time
import json
import sys
from pathlib import Path

ablation_dir = Path(r"e:\Research Projects\706\outputs\ablations\20260423_032158")
results_file = ablation_dir / "results.csv"
main_log = ablation_dir / "ablation_runner.log"
status_file = Path(r"e:\Research Projects\706\ablation_status.json")

print("Starting monitor process...")
print(f"Checking for completion of ablation runner")
print(f"Results file: {results_file}")

max_wait_minutes = 240  # 4 hour timeout max
check_interval = 30    # Check every 30 seconds

start_time = time.time()
last_cell_count = 0

while True:
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    
    if elapsed_min > max_wait_minutes:
        print(f"\nMax wait time ({max_wait_minutes}min) exceeded. Saving status and exiting.")
        break
    
    # Check if results file exists (completion indicator)
    if results_file.exists():
        print(f"\n✓ RUNNER COMPLETED!")
        with open(results_file) as f:
            csv_content = f.read()
        
        # Count cells
        cell_count = csv_content.count('\n') - 1  # Subtract header
        
        status = {
            "status": "COMPLETED",
            "elapsed_minutes": elapsed_min,
            "cells_completed": cell_count,
            "output_dir": str(ablation_dir),
            "timestamp": time.time()
        }
        break
    
    # Check log for current progress
    if main_log.exists():
        with open(main_log) as f:
            log_content = f.read()
        
        # Count "success=" entries
        success_count = log_content.count("success=")
        if success_count > last_cell_count:
            last_cell_count = success_count
            # Extract latest cell info
            for line in reversed(log_content.split('\n')):
                if "[" in line and "/24]" in line:
                    print(f"[{elapsed_min:.1f}min] {line.strip()[:90]}")
                    break
    
    time.sleep(check_interval)

# Save status
with open(status_file, 'w') as f:
    json.dump(status, f, indent=2)

print(f"\nStatus saved to: {status_file}")
print(f"Total elapsed: {elapsed_min:.1f} minutes")
if results_file.exists():
    print(f"Cells completed: {cell_count}")
    print(f"Results CSV: {results_file}")
    
    # Print summary from CSV
    with open(results_file) as f:
        lines = f.readlines()
    if len(lines) > 1:
        print(f"\nFirst result row:")
        print(lines[1][:80])
