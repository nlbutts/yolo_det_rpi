# YOLOv8 ONNX Detection & ZMQ Server

This project provides a highly optimized YOLOv8n object detection implementation using OpenCV DNN, a ZeroMQ-based inference server, and a cross-compilation environment for Raspberry Pi 4.

## 🚀 Performance Optimizations
- **NumPy Vectorization**: Post-processing is fully vectorized, reducing inference latency by orders of magnitude compared to pure Python loops.
- **Multithreading**: Configurable thread count for OpenCV CPU inference.
- **Minimal OpenCV**: Scripts and Docker environment provided to build a stripped-down version of OpenCV (80-90% smaller) specifically for ARM64.

---

## 🛠 Features

### 1. Standalone Detection (`yolo_detect.py`)
Run inference on a single image.
```bash
python3 yolo_detect.py --model yolov8n.onnx --image human.jpg --runs 5 --threads 4
```
- `--runs`: Runs detection N times, ignores the first "warm-up" run, and outputs the average timing.
- `--threads`: Sets the number of CPU threads for OpenCV.

### 2. ZMQ Inference Server
A Request-Reply (REQ-REP) architecture for remote inference.

**Start the Server:**
```bash
python3 yolo_server.py --model yolov8n.onnx --port 5555
```

**Run the Client:**
```bash
python3 yolo_client.py --image human.jpg --server tcp://localhost:5555
```

---

## 🥧 Raspberry Pi 4 Cross-Compilation

We use Docker with QEMU emulation to build a minimal version of OpenCV for ARM64 on an x86 host. This build includes only `core`, `imgproc`, `imgcodecs`, and `dnn`.

### Requirements
- Docker
- `binfmt-support` and `qemu-user-static` (to run ARM containers on x86)

### How to Build
To build for Raspberry Pi 4 (64-bit), use the `--platform linux/arm64` flag:
```bash
# Register QEMU (run once on host)
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Build the ARM64 container
docker build -t opencv-arm64-builder --platform linux/arm64 .

# Run the build script in the ARM64 environment
docker run --rm --platform linux/arm64 -v $(pwd):/workspace opencv-arm64-builder
```
The output will be placed in `./opencv_minimal_install/wheel/`. You can copy this `.whl` file directly to your Raspberry Pi.

---

## 📦 Dependencies
- `opencv-python`
- `numpy`
- `pyzmq` (for server/client)

Install via:
```bash
pip install opencv-python numpy pyzmq
```

## 📄 License
MIT
