# 火灾检测模型人工修正与迭代组件

本组件是一个独立的代码库，旨在为火灾检测模型提供人工校验、数据修正以及自动化的模型迭代功能。

## 主要功能
- **自动扫描**：实时监听 `runs/` 目录下的新输出，解析模型预测的图片与标注文件。
- **图形化校验**：基于 PyQt6 的 GUI 界面，支持点击缩略图查看大图、拖动修改检测框、一键确认/否认火灾。
- **数据导出**：支持将校验后的数据导出为标准 COCO 或 Pascal VOC 格式。
- **自动再训练**：支持迁移学习(Yolo)，自动调参 (Optuna)，早停机制，并集成 TensorBoard 可视化。
- **模型验证与回滚**：仅当新模型 mAP@0.5 提升 ≥ 2% 时才覆盖原模型，否则自动回滚。

## 环境要求
- Python 3.8+
- Raspberry Pi OS / Ubuntu 20.04+ / Windows 10
- 推荐使用虚拟环境（如 `venv` 或 `conda`）

## 快速开始

### 1. 一键启动 (推荐用于树莓派/Linux)
在 Linux (如 Raspberry Pi OS 5) 下，建议先安装系统级库以避免编译错误（特别是 Pillow、PyQt6 和 OpenCV）：
```bash
sudo apt update
sudo apt install python3-pyqt6 python3-opencv python3-pil python3-venv
```
然后执行脚本，它将自动创建虚拟环境并引用上述系统库：
```bash
chmod +x start.sh
./start.sh
```

### 2. 手动安装 (Windows/其他)
```bash
pip install -r requirements.txt
python main.py
```

### 3. 配置说明
您可以直接在程序中点击 **"Settings"** 按钮来修改配置，也可以手动编辑 `config.yaml`：
- `input.base_dir`: 模型输出文件夹路径（默认为 `../runs`）。
- `training.base_model_path`: 原始权重文件路径。
- `gui.reviewer_list`: 审核人员名单。

### 4. 运行测试
```bash
pytest tests/ --cov=core --cov-report=term-missing
```

## 目录结构
- `core/`: 核心逻辑（数据库、扫描器、导出器、训练器）。
- `ui/`: 图形界面实现。
- `tests/`: 单元测试。
- `main.py`: 程序主入口。
- `config.yaml`: 配置文件。
- `corrections.sqlite`: 标注结果数据库。

## 再训练流程
1. 在 GUI 中完成对预测结果的人工修正。
2. 点击 "Export Dataset" 导出标注好的数据集。
3. 点击 "Retrain Model" 启动自动化再训练流程。
4. 训练完成后，程序会自动对比 mAP，符合条件则更新主项目权重。

