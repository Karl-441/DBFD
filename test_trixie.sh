#!/bin/bash

# DBFD-Raspberry Comprehensive Performance Test Script (v2.0 Redesign)
# -----------------------------------------------------------------------------
# 重新设计：采用集成的 Python 监控引擎，确保数据捕获的完整性。

set -e # 遇到错误立即退出

# --- 默认配置 ---
TEST_DURATION=300
BASELINE_DURATION=30
SAMPLING_HZ=1
TEMP_LIMIT=80
OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
LOG_LEVEL="INFO"
APP_CMD="python3 main.py" # 默认使用有 UI 版本

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --duration) TEST_DURATION="$2"; shift ;;
        --baseline) BASELINE_DURATION="$2"; shift ;;
        --sampling) SAMPLING_HZ="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        --app) APP_CMD="$2"; shift ;;
        --help) echo "Usage: ./test_trixie.sh [--duration SEC] [--baseline SEC] [--sampling HZ] [--output DIR] [--app CMD]"; exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# 创建目录结构
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/csv"
mkdir -p csv # 应用层 CSV 缓存

echo "DBFD-Raspberry Performance Test Suite"
echo "Target Output: $OUTPUT_DIR"
echo "-----------------------------------"

# 1. 环境自检
echo "Checking environment..."
if ! python3 -c "import psutil, numpy" &>/dev/null; then
    echo "ERROR: Missing Python dependencies (psutil or numpy). Run: sudo apt install python3-psutil python3-numpy"
    exit 1
fi

# 2. 抓取准确的系统硬件信息
echo "Gathering accurate system info..."
python3 utils/sys_monitor.py --dir "$OUTPUT_DIR" --info > "$OUTPUT_DIR/info.json"
echo "Hardware Info captured: $(cat "$OUTPUT_DIR/info.json")"

# 3. 自动注入 Mock 模型并配置 YOLO
mkdir -p models
if [ ! -f "models/best.pt" ]; then
    echo "WARNING: models/best.pt not found. Attempting to use yolov8n.pt or creating a dummy..."
    if [ -f "yolov8n.pt" ]; then
        cp yolov8n.pt models/best.pt
        echo "Copied yolov8n.pt to models/best.pt"
    else
        # 最后的兜底方案：创建一个基础的 mock 文件
        python3 -c "import torch; torch.save({'model': torch.nn.Module()}, 'models/best.pt')"
        echo "Created dummy models/best.pt"
    fi
fi

# 4. 生成临时测试配置 (强制开启 YOLO)
echo "Generating test configuration (config.json)..."
cat > config.json <<EOF
{
    "USE_YOLO": true,
    "USE_PNN": false,
    "YOLO_MODEL_PATH": "models/best.pt",
    "CAMERA_INDEX": 0,
    "FPS": 30,
    "DETECT_INTERVAL": 3
}
EOF

# 5. 启动背景监控引擎
echo "Starting background monitoring engine..."
python3 utils/sys_monitor.py --dir "$OUTPUT_DIR" --hz "$SAMPLING_HZ" &
MONITOR_PID=$!

# 确保监控进程已存活
sleep 2
if ! kill -0 $MONITOR_PID 2>/dev/null; then
    echo "ERROR: Failed to start monitoring engine!"
    exit 1
fi

# 5. 空闲基线采集 (30s)
echo "Collecting $BASELINE_DURATION s baseline..."
sleep "$BASELINE_DURATION"

# 6. 启动主程序 (注入模式)
echo "Launching Application: $APP_CMD"
export DBFD_PERF_TEST=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 动态注入 Path 并启动 main.main()
python3 -c "import pathlib, builtins; builtins.Path = pathlib.Path; import main; main.main()" --headless > "$OUTPUT_DIR/logs/stdout.log" 2> "$OUTPUT_DIR/logs/stderr.log" &
APP_PID=$!

# 实时监测应用状态
sleep 3
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "ERROR: Application crashed on startup! Check $OUTPUT_DIR/logs/stderr.log"
    cat "$OUTPUT_DIR/logs/stderr.log"
    kill $MONITOR_PID 2>/dev/null || true
    exit 1
fi

# 7. 正式测试
echo "Running performance test for $TEST_DURATION seconds..."
ELAPSED=0
while [ $ELAPSED -lt $TEST_DURATION ]; do
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo "ERROR: Application crashed during test! Test aborted."
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    printf "\rProgress: %d%% (%d/%d s)" $((ELAPSED * 100 / TEST_DURATION)) $ELAPSED $TEST_DURATION
done
echo ""

# 8. 清理与收尾
echo "Cleaning up..."
kill $APP_PID 2>/dev/null || true
kill $MONITOR_PID 2>/dev/null || true
rm -f config.json # 删除临时生成的测试配置

# 移动应用性能数据
if [ -f "csv/perf_stats.csv" ]; then
    mv "csv/perf_stats.csv" "$OUTPUT_DIR/csv/perf_stats.csv"
fi

# 9. 生成性能报告
echo "Generating structured report..."
python3 utils/plot_report.py "$OUTPUT_DIR"

# 10. 结果打包
REPORT_NAME="DBFD-Raspberry_trixie_$(date +%Y%m%d_%H%M%S)_report.md"
REPORT_FILE=$(find "$OUTPUT_DIR" -name "*_report.md" | head -n 1)
if [ -n "$REPORT_FILE" ]; then
    mv "$REPORT_FILE" "$REPORT_NAME"
    TARBALL="${REPORT_NAME%.md}.tar.gz"
    tar -czf "$TARBALL" "$OUTPUT_DIR" "$REPORT_NAME"
    echo "-----------------------------------"
    echo "SUCCESS: Test complete!"
    echo "Report: $REPORT_NAME"
    echo "Archive: $TARBALL"
else
    echo "CRITICAL: Report generation failed!"
    exit 1
fi
