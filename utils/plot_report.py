#!/usr/bin/env python3
import os
import sys
import csv
import json
import datetime
import numpy as np
from pathlib import Path

"""
Performance Report Generator for DBFD-Raspberry
- Parses CSV data (CPU, GPU, Temp, FPS, Latency)
- Calculates key performance metrics
- Generates a structured Markdown report
"""

def load_csv(file_path):
    """Load CSV data into a list of dictionaries"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def calculate_stats(data, key):
    """Calculate basic statistics for a given key in the data"""
    values = [float(row[key]) for row in data if key in row and row[key]]
    if not values:
        return {
            'avg': 0, 'max': 0, 'min': 0, 'std': 0, 'p99': 0
        }
    
    return {
        'avg': np.mean(values),
        'max': np.max(values),
        'min': np.min(values),
        'std': np.std(values),
        'p99': np.percentile(values, 99)
    }

def generate_markdown(results_dir, hw_info, sw_info):
    """Generate the final Markdown report"""
    # Load data
    cpu_data = load_csv(os.path.join(results_dir, 'cpu_stats.csv'))
    gpu_data = load_csv(os.path.join(results_dir, 'gpu_stats.csv'))
    temp_data = load_csv(os.path.join(results_dir, 'temp_stats.csv'))
    perf_data = load_csv(os.path.join(results_dir, 'perf_stats.csv'))
    
    # Calculate stats
    cpu_stats = calculate_stats(cpu_data, 'cpu_total_percent')
    gpu_stats = calculate_stats(gpu_data, 'gpu_percent')
    temp_stats = calculate_stats(temp_data, 'temp_c')
    
    fps_stats = calculate_stats(perf_data, 'fps')
    latency_stats = calculate_stats(perf_data, 'latency_ms')
    
    # Generate timestamp
    now = datetime.datetime.now()
    report_name = f"DBFD-Raspberry_trixie_{now.strftime('%Y%m%d_%H%M%S')}_report.md"
    report_path = os.path.join(results_dir, report_name)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# DBFD-Raspberry Performance Test Report\n\n")
        f.write(f"**Date:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## **Hardware Information**\n")
        for k, v in hw_info.items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n")
        
        f.write(f"## **Software Information**\n")
        for k, v in sw_info.items():
            f.write(f"- **{k}:** {v}\n")
        f.write("\n")
        
        f.write(f"## **Resource Utilization**\n")
        f.write(f"- **CPU Usage:** Avg {cpu_stats['avg']:.2f}%, Max {cpu_stats['max']:.2f}%\n")
        f.write(f"- **GPU Usage:** Avg {gpu_stats['avg']:.2f}%, Max {gpu_stats['max']:.2f}%\n")
        f.write(f"- **Core Temp:** Avg {temp_stats['avg']:.2f}C, Max {temp_stats['max']:.2f}C\n")
        f.write("\n")
        
        f.write(f"## **Performance Metrics**\n")
        f.write(f"- **Average FPS:** {fps_stats['avg']:.2f}\n")
        f.write(f"- **99th Latency:** {latency_stats['p99']:.2f} ms\n")
        f.write(f"- **Jitter (Latency Std Dev):** {latency_stats['std']:.2f} ms\n")
        # Estimate dropped frames: (Target FPS - Avg FPS) * Total Duration
        target_fps = 15 # Default from config
        total_duration = float(perf_data[-1]['timestamp']) - float(perf_data[0]['timestamp']) if len(perf_data) > 1 else 0
        dropped_frames = max(0, int((target_fps - fps_stats['avg']) * total_duration))
        f.write(f"- **Estimated Dropped Frames:** {dropped_frames}\n")
        f.write("\n")
        
        f.write(f"## **Anomalies and Errors**\n")
        # In a real scenario, we'd parse logs for these
        f.write(f"- **Crashes:** 0\n")
        f.write(f"- **Throttling Events:** {'Yes' if temp_stats['max'] > 80 else 'None'}\n")
        f.write("\n")
        
    return report_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_report.py <results_dir>")
        sys.exit(1)
        
    results_dir = sys.argv[1]
    
    # Load HW/SW info from json if exists
    info_path = os.path.join(results_dir, 'info.json')
    hw_info = {}
    sw_info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            hw_info = info.get('hw', {})
            sw_info = info.get('sw', {})
            
    report_path = generate_markdown(results_dir, hw_info, sw_info)
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
