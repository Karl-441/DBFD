# Memory Optimization Report for DBFD-Raspberry

## 1. Analysis of Memory Usage

### Baseline
- **Original Resolution**: 640x480
- **Original FPS**: 15
- **Original Threshold**: 800MB
- **Observed Issues**: 
  - OpenCV frame buffers accumulating.
  - Python GC lazy collection causing RSS drift.
  - GLib/malloc fragmentation on Linux.

### Key Bottlenecks
1. **Frame Buffers**: Each 640x480x3 frame takes ~900KB. With buffering and copies (preprocess, masking), this grows quickly.
2. **Import Overhead**: `PyQt6` adds ~50-100MB overhead even if not fully used. Headless mode avoids this.
3. **Fragmentation**: Long-running Python processes on Linux often suffer from memory fragmentation where released memory isn't returned to the OS.

## 2. Optimization Measures Implemented

### 2.1 Configuration Tuning (`config.py`)
- **Resolution Reduced**: 640x480 -> **320x240**.
  - *Impact*: 4x reduction in pixel count and memory per frame.
  - *Justification*: Fire detection features (color/texture) are robust enough at lower resolutions.
- **FPS Reduced**: 15 -> **10**.
  - *Impact*: Reduced CPU load gives more time for GC and cleanup between frames.
- **Memory Limit**: Set to **400MB**.

### 2.2 Code Optimization (`headless_runner.py`)
- **Explicit Deletion**: Added `del frame` and `del vis` to ensure reference counts drop immediately.
- **Periodic GC**: Added `gc.collect()` every 100 frames to force cycle collection.

### 2.3 System/Runtime Tuning
- **malloc_trim (`memory_monitor.py`)**: 
  - Uses `ctypes` to call `libc.malloc_trim(0)`.
  - *Effect*: Forces the allocator to release free heap memory back to the OS.
- **MALLOC_ARENA_MAX (`start_service.sh`)**:
  - Set to `2`.
  - *Effect*: Limits the number of memory arenas, significantly reducing fragmentation on multi-core systems like Pi 4.

## 3. Verification Plan

### Test Scenario
1. Start `python3 main.py --headless`.
2. Monitor RSS usage using `htop` or the built-in logger.
3. Simulate fire detection (show a red/orange object to camera) to trigger processing pipelines.
4. Run for 1 hour.

### Expected Results
- **Idle Memory**: ~60-80MB
- **Active Detection Memory**: ~120-180MB
- **Peak Memory**: < 250MB
- **Stability**: No upward drift in RSS over time.

## 4. Conclusion
With these changes, the application is expected to comfortably stay within the 400MB limit, likely averaging under 200MB in headless mode.
