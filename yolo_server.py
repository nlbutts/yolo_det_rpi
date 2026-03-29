import argparse
import time
import struct
import zmq
import numpy as np
import cv2
import json
from yolo_detect import COCODetector

def main():
    parser = argparse.ArgumentParser(description="YOLO ZMQ Server")
    parser.add_argument("-m", "--model", required=True, help="Path to YOLOv8n ONNX model file")
    parser.add_argument("-p", "--port", type=int, default=5555, help="ZMQ port to listen on")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("-n", "--nms-thresh", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("-t", "--threads", type=int, default=0, help="Number of threads for OpenCV")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    detector = COCODetector(
        args.model,
        confidence_thresh=args.confidence,
        nms_thresh=args.nms_thresh,
        num_threads=args.threads,
    )
    print("Model loaded successfully!")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"ZMQ Server listening on port {args.port}...")

    while True:
        try:
            # Wait for next request from client
            # Expecting a multipart message: [header_bytes, image_bytes]
            message = socket.recv_multipart()

            if len(message) != 2:
                socket.send_json({"error": "Invalid message format. Expected 2 parts: header and image data."})
                continue

            header, image_data = message

            if len(header) != 8:
                socket.send_json({"error": "Invalid header size. Expected 8 bytes."})
                continue

            width, height = struct.unpack('<II', header)
            expected_size = width * height * 3

            if len(image_data) != expected_size:
                socket.send_json({"error": f"Invalid image data size. Expected {expected_size}, got {len(image_data)}."})
                continue

            # Convert to numpy array
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))

            # Run inference
            start = time.perf_counter()
            _, detections = detector.detect(image)
            end = time.perf_counter()

            duration_ms = (end - start) * 1000
            print(f"Processed {width}x{height} image in {duration_ms:.2f}ms. Found {len(detections)} object(s).")

            response = {
                "success": True,
                "inference_time_ms": duration_ms,
                "detections": detections
            }
            socket.send_json(response)

        except KeyboardInterrupt:
            print("\nShutting down server...")
            break
        except Exception as e:
            print(f"Error processing request: {e}")
            try:
                socket.send_json({"error": str(e)})
            except Exception:
                pass

if __name__ == "__main__":
    main()
