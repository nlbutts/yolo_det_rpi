#!/usr/bin/env python3
"""YOLOv8 inference using LiteRT with ArmNN delegate."""

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import cv2

from ai_edge_litert import interpreter

# Set LD_LIBRARY_PATH for ArmNN libraries
armnn_path = "/home/nlbutts/projects/litert/armnn"
delegate_path = "/home/nlbutts/projects/litert/armnn/delegate"
if "ARMNN_RESTARTED" not in os.environ:
    os.environ["ARMNN_RESTARTED"] = "1"
    os.environ["LD_LIBRARY_PATH"] = f"{armnn_path}:{delegate_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


# COCO class labels (80 classes)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_image(
    image_path: str, target_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """Load and preprocess image for YOLOv8 inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_array[np.newaxis, ...]


def box_xywh_to_xyxy(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert bbox from xywh to xyxy format."""
    x_min = (x - w / 2) * img_w
    y_min = (y - h / 2) * img_h
    x_max = (x + w / 2) * img_w
    y_max = (y + h / 2) * img_h
    return (
        float(np.clip(x_min, 0, img_w)),
        float(np.clip(y_min, 0, img_h)),
        float(np.clip(x_max, 0, img_w)),
        float(np.clip(y_max, 0, img_h)),
    )


def non_max_suppression(
    detections: List[Tuple[float, float, float, float, float, str]],
    iou_threshold: float,
) -> List[Tuple[float, float, float, float, float, str]]:
    """Apply Non-Maximum Suppression to filter overlapping detections."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        if not detections:
            break

        best_xmin, best_ymin, best_xmax, best_ymax = best[:4]
        best_area = (best_xmax - best_xmin) * (best_ymax - best_ymin)

        remaining = []
        for det in detections:
            det_xmin, det_ymin, det_xmax, det_ymax = det[:4]

            x_overlap = max(0, min(best_xmax, det_xmax) - max(best_xmin, det_xmin))
            y_overlap = max(0, min(best_ymax, det_ymax) - max(best_ymin, det_ymin))
            intersection = x_overlap * y_overlap

            det_area = (det_xmax - det_xmin) * (det_ymax - det_ymin)
            iou = intersection / (best_area + det_area - intersection)

            if iou < iou_threshold:
                remaining.append(det)

        detections = remaining

    return keep


def post_process_yolo_output(
    output: np.ndarray,
    img_w: int,
    img_h: int,
    conf_threshold: float,
    iou_threshold: float,
) -> List[Tuple[float, float, float, float, float, str]]:
    """Post-process YOLOv8 output to extract detections."""
    output_flat = output[0]

    x = output_flat[0, :]
    y = output_flat[1, :]
    w = output_flat[2, :]
    h = output_flat[3, :]

    class_scores = output_flat[4:, :]

    max_class_scores = class_scores.max(axis=0)
    max_class_ids = class_scores.argmax(axis=0)

    valid_idx = np.where(max_class_scores > conf_threshold)[0]

    detections = []
    for idx in valid_idx:
        bbox_x, bbox_y, bbox_w, bbox_h = x[idx], y[idx], w[idx], h[idx]
        conf = max_class_scores[idx]
        class_id = max_class_ids[idx]
        class_name = COCO_CLASSES[class_id]

        x_min, y_min, x_max, y_max = box_xywh_to_xyxy(
            bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h
        )
        detections.append((x_min, y_min, x_max, y_max, conf, class_name))

    detections = non_max_suppression(detections, iou_threshold)

    return detections


class COCODetector:
    """Class to encapsulate YOLOv8 inference using LiteRT with ArmNN delegate."""

    def __init__(
        self,
        model_path: str,
        confidence_thresh: float = 0.5,
        nms_thresh: float = 0.45,
        num_threads: int = None,
        delegate_path: str = "/home/nlbutts/projects/litert/armnn/delegate/libarmnnDelegate.so.29.1",
    ):
        self.model_path = model_path
        self.conf_threshold = confidence_thresh
        self.iou_threshold = nms_thresh
        self.threads = num_threads

        print(f"Initializing COCODetector with model: {model_path}")
        print(f"Loading delegate: {delegate_path}")

        self.delegate = None
        if "int8" in model_path.lower():
            print(
                "Warning: YOLOv8 INT8 model detected. Falling back to XNNPACK for better performance/precision."
            )
        else:
            try:
                delegate_options = {
                    "backends": "CpuAcc,GpuAcc,CpuRef",
                    "enable-fast-math": "1",
                    "logging-severity": "warning",
                }
                self.delegate = interpreter.load_delegate(
                    delegate_path, options=delegate_options
                )
                print("Delegate loaded successfully")
            except Exception as e:
                print(f"Failed to load ArmNN delegate: {e}")
                print("Falling back to LiteRT with XNNPACK...")

        if self.delegate:
            self.interpreter = interpreter.Interpreter(
                model_path=model_path,
                experimental_delegates=[self.delegate],
                num_threads=self.threads,
            )
        else:
            self.interpreter = interpreter.Interpreter(
                model_path=model_path, num_threads=self.threads
            )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

    def detect(self, image: np.ndarray):
        """Run YOLOv8 inference on a BGR numpy array from OpenCV."""
        img_h, img_w = image.shape[:2]

        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = input_data[np.newaxis, ...]

        # Inference
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)

        # Post-process
        raw_detections = post_process_yolo_output(
            output, img_w, img_h, self.conf_threshold, self.iou_threshold
        )

        # Format for server consumption
        formatted_detections = []
        for d in raw_detections:
            formatted_detections.append(
                {
                    "bbox": [d[0], d[1], d[2], d[3]],
                    "confidence": float(d[4]),
                    "class_name": d[5],
                }
            )

        return output, formatted_detections


def run_inference(
    model_path: str,
    image_path: str,
    num_iterations: int,
    conf_threshold: float,
    iou_threshold: float,
    delegate_path: str,
    threads: int,
) -> None:
    """Run YOLOv8 inference on an image using ArmNN delegate."""
    detector = COCODetector(
        model_path=model_path,
        confidence_thresh=conf_threshold,
        nms_thresh=iou_threshold,
        num_threads=threads,
        delegate_path=delegate_path,
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    times = []

    for i in range(num_iterations):
        start_time = time.perf_counter()
        output, formatted_detections = detector.detect(img)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    print(f"\nModel: {model_path}")
    print(f"Image: {image_path}")
    print(f"Iterations: {num_iterations}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Delegate: {'ArmNN' if detector.delegate else 'None (XNNPACK)'}")
    print(f"\nInference times:")
    print(f"  Min: {min(times) * 1000:.2f} ms")
    print(f"  Max: {max(times) * 1000:.2f} ms")
    print(f"  Avg: {np.mean(times) * 1000:.2f} ms")
    print(f"  FPS: {1 / np.mean(times):.2f}")

    print(f"\nDetections ({len(formatted_detections)} found):")
    if formatted_detections:
        for i, det in enumerate(formatted_detections, 1):
            x_min, y_min, x_max, y_max = det["bbox"]
            print(
                f"  {i}. {det['class_name']} ({det['confidence'] * 100:.1f}%) "
                f"[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]"
            )
    else:
        print("  No detections found above confidence threshold")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference using LiteRT with ArmNN delegate"
    )
    parser.add_argument(
        "-m","--model", type=str, required=True, help="Path to the YOLOv8 TFLite model"
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations (default: 1)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--delegate",
        type=str,
        default="/home/nlbutts/projects/litert/armnn/delegate/libarmnnDelegate.so.29.1",
        help="Path to the ArmNN delegate library",
    )

    parser.add_argument(
        "-t", "--num-threads",
        type=int,
        default=None,
        help="Number of threads for CPU execution (default: auto)",
    )

    args = parser.parse_args()
    run_inference(
        args.model,
        args.image,
        args.iterations,
        args.conf_threshold,
        args.iou_threshold,
        args.delegate,
        args.num_threads
    )


if __name__ == "__main__":
    main()
