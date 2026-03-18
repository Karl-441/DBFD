# DBFD-Raspberry Performance Test Suite

一套专为树莓派 (Raspberry Pi OS Trixie) 设计的综合性能测试工具，用于评估 `DBFD-Raspberry` 项目的运行效率、稳定性和资源占用。

## 核心功能

1.  **多维度数据采集**:
    *   **CPU**: 记录 User/System/Idle/IOWait 占用率及总占用。
    *   **GPU**: 记录 V3D 频率（通过 `vcgencmd`）。
    *   **温度**: 实时监控 SoC 核心温度，具备 80°C 自动停机保护。
    *   **应用性能**: 自动采集 FPS、端到端延迟（Latency）和抖动（Jitter）。
2.  **自动化流程**:
    *   30 秒空闲基线采集。
    *   5 分钟（可配置）主程序无头模式运行。
    *   日志自动滚动切分（单文件 50MB 限制）。
3.  **结构化报告**:
    *   自动生成 Markdown 格式的详细性能报告。
    *   打包所有 CSV 原始数据和日志为 `.tar.gz`。

## 文件结构

*   `test_trixie.sh`: 主测试脚本（Bash）。
*   `conf/test_config.yaml`: 测试参数配置。
*   `utils/plot_report.py`: 报告生成与数据分析脚本。
*   `README.md`: 本说明文件。

## 使用方法

### 1. 准备环境
确保已安装必要依赖：
```bash
sudo apt update
sudo apt install python3-psutil python3-numpy bc vcgencmd
```

### 2. 运行测试
在项目根目录下执行：
```bash
chmod +x test_trixie.sh
./test_trixie.sh
```

### 3. 自定义参数
支持通过命令行覆盖默认配置：
```bash
./test_trixie.sh --duration 600 --sampling 2 --output my_results
```
*   `--duration`: 测试时长（秒），默认 300。
*   `--baseline`: 基线采集时长（秒），默认 30。
*   `--sampling`: 采样频率（Hz），默认 1。
*   `--output`: 输出目录名称。

## 验收标准
*   **重复性**: 连续两次测试的关键指标误差应 ≤ 3%。
*   **安全性**: 温度超过 80°C 时脚本将立即终止应用以保护硬件。
*   **完整性**: 每次运行都会生成 `DBFD-Raspberry_trixie_YYYYMMDD_HHMMSS_report.md` 及对应的压缩包。

## 后续优化
测试报告将作为性能优化的直接依据。根据报告中的 CPU/GPU 瓶颈、延迟分布和热点函数，我们将实施针对性优化（如算法加速、多线程优化等），并使用本脚本进行回归验证。
