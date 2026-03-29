import argparse
import time
import struct
import zmq
import cv2
import json

def main():
    parser = argparse.ArgumentParser(description="YOLO ZMQ Client")
    parser.add_argument("-i", "--image", required=True, help="Path to input image file")
    parser.add_argument("-s", "--server", default="tcp://localhost:5555", help="ZMQ server address")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    height, width, channels = image.shape
    if channels != 3:
        print(f"Error: Image must have 3 channels, got {channels}.")
        return

    print(f"Loaded image {args.image}: {width}x{height}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    print(f"Connecting to server at {args.server}...")
    socket.connect(args.server)

    # Prepare message body
    # Header format: struct.pack('<II', width, height)
    header = struct.pack('<II', width, height)
    image_bytes = image.tobytes()

    print("Sending request to server...")
    start_total = time.perf_counter()
    socket.send_multipart([header, image_bytes])

    # Wait for response
    response_bytes = socket.recv()
    end_total = time.perf_counter()

    response = json.loads(response_bytes.decode('utf-8'))
    total_time_ms = (end_total - start_total) * 1000

    print(f"\nRoundtrip total time: {total_time_ms:.2f}ms")

    if response.get("success"):
        print(f"Server inference time: {response['inference_time_ms']:.2f}ms")
        detections = response.get("detections", [])
        
        if not detections:
            print("No detections found.")
        else:
            print(f"\nDetected {len(detections)} object(s):")
            print(f"{'Class':<15}{'Confidence':<12}{'Bounding Box':<25}")
            print("-" * 52)
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                print(
                    f"{det['class_name']:<15}{det['confidence']:<12.3f}"
                    f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )
    else:
        print(f"Error from server: {response.get('error')}")

if __name__ == "__main__":
    main()
