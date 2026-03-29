#!/usr/bin/env python3
"""YOLOv8n ONNX Inference Program using OpenCV DNN"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class COCODetector:
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

    def __init__(self, model_path, confidence_thresh=0.5, nms_thresh=0.45, num_threads=0):
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        
        if num_threads > 0:
            cv2.setNumThreads(num_threads)
            
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU_FP16)
        
        self.input_width = 640
        self.input_height = 640

    def detect(self, image_input):
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input

        original_height, original_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 255.0,
            (self.input_width, self.input_height),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward()

        detections = self._post_process(outputs, original_height, original_width)

        return image, detections

    def _post_process(self, outputs, height, width):
        scale_x = width / self.input_width
        scale_y = height / self.input_height

        boxes = []
        confidences = []
        class_ids_list = []

        # outputs[0] for YOLOv8 is usually (1, 84, 8400) or similar
        detections = outputs[0]
        if detections.ndim == 3:
            detections = detections[0]
        
        # Transpose to (8400, 84) where each row is [x, y, w, h, cls0...cls79]
        if detections.shape[0] < detections.shape[1]:
            detections = detections.T
            
        # Extract fields
        bboxes = detections[:, :4]
        scores = detections[:, 4:]
        
        # Vectorized max score and class ID
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Mask valid detections
        mask = max_scores > self.confidence_thresh
        
        valid_bboxes = bboxes[mask]
        valid_confidences = max_scores[mask]
        valid_class_ids = class_ids[mask]
        
        for i in range(len(valid_bboxes)):
            bbox = valid_bboxes[i]
            # Convert [center_x, center_y, width, height] -> [x_min, y_min, width, height]
            x_center, y_center, w, h = bbox
            x_min = (x_center - w / 2) * scale_x
            y_min = (y_center - h / 2) * scale_y
            bw = w * scale_x
            bh = h * scale_y
            
            boxes.append([x_min, y_min, bw, bh])
            confidences.append(float(valid_confidences[i]))
            class_ids_list.append(int(valid_class_ids[i]))
            
        if not boxes:
            return []
            
        class_ids = class_ids_list

        boxes_array = np.array(boxes, dtype=np.float32)
        confidences_array = np.array(confidences, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_array, confidences_array, self.confidence_thresh, self.nms_thresh
        )

        results = []
        for i in indices:
            box = boxes[i]
            x1, y1 = float(box[0]), float(box[1])
            x2, y2 = float(x1 + box[2]), float(y1 + box[3])
            cls_id = class_ids[i]
            class_name = (
                self.COCO_CLASSES[cls_id]
                if cls_id < len(self.COCO_CLASSES)
                else f"class_{cls_id}"
            )
            results.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidences[i],
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n ONNX Inference")
    parser.add_argument(
        "-m", "--model", required=True, help="Path to YOLOv8n ONNX model file"
    )
    parser.add_argument("-i", "--image", required=True, help="Path to input image file")
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "-n",
        "--nms-thresh",
        type=float,
        default=0.45,
        help="NMS IoU threshold (0.0-1.0)",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of times to run detection (default: 1)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="Number of threads for OpenCV CPU inference (default: 0, uses all available)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("Error: confidence must be between 0.0 and 1.0")
        return 1

    if not 0.0 <= args.nms_thresh <= 1.0:
        print("Error: nms-thresh must be between 0.0 and 1.0")
        return 1

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1

    print(f"\nLoading model: {model_path}")
    print(f"Loading image: {image_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS threshold: {args.nms_thresh}\n")

    print("Loading YOLOv8 model...")
    detector = COCODetector(
        model_path, 
        confidence_thresh=args.confidence, 
        nms_thresh=args.nms_thresh,
        num_threads=args.threads
    )
    print("Model loaded successfully!\n")

    image = cv2.imread(str(image_path))
    print(f"Image size: {image.shape[1]}x{image.shape[0]}\n")

    print(f"Running inference {args.runs} time(s)...")
    
    durations = []
    detections = []
    
    for i in range(args.runs):
        start = time.perf_counter()
        _, detections = detector.detect(image_path)
        end = time.perf_counter()
        
        duration_ms = (end - start) * 1000
        durations.append(duration_ms)
        
        if args.runs > 1:
            print(f"Run {i+1}/{args.runs}: {duration_ms:.2f}ms")

    if args.runs > 1:
        # Ignore the first run for averaging if N > 1
        avg_duration = (sum(durations) - durations[0]) / (args.runs - 1)
        print(f"\nAverage inference time (excluding 1st run): {avg_duration:.2f}ms")
    else:
        print(f"\nInference time: {durations[0]:.2f}ms")
    print()

    if not detections:
        print("No detections found above confidence threshold.")
    else:
        print(f"Detected {len(detections)} object(s):\n")
        print(f"{'Class':<15}{'Confidence':<12}{'Bounding Box':<25}")
        print("-" * 52)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"{det['class_name']:<15}{det['confidence']:<12.3f}"
                f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

    return 0


if __name__ == "__main__":
    exit(main())
