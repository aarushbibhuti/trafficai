import os
from collections import defaultdict, deque
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Global list to store clicked pixel coordinates
clicked_points = []

def click_event(event, x, y, flags, params):
    """Callback function to capture mouse clicks on the calibration frame."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        # Draw a small circle and label on the frame to show the selection
        cv2.circle(params['frame'], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(params['frame'], f"P{len(clicked_points)}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Click 4 Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left", params['frame'])


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(reshaped_points, self.m).reshape(-1, 2)


def main():
    # --- 1. Get Region Standards for Normalization ---
    print("--- TrafficAI Perspective Calibration ---")
    region = input("Are you analyzing US or EU roads? (us/eu): ").strip().lower()
    
    # Standard dimensions in meters for a single lane segment:
    # US standard: Lane width ~3.7m, full skip-line cycle (line + gap) ~12.2m
    # EU standard: Lane width ~3.5m, full skip-line cycle ~12.0m
    if region == "eu":
        TARGET_WIDTH = 3.5
        TARGET_HEIGHT = 12.0
        unit_label = "km/h"
        speed_multiplier = 3.6
    else:  # Default to US
        TARGET_WIDTH = 3.7
        TARGET_HEIGHT = 12.2
        unit_label = "MPH"
        speed_multiplier = 2.23694

    # --- 2. File and Path Setup ---
    input_video = "data/videos/newtraffic.mkv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.abspath(os.path.join("data/processed", f"track_{timestamp}"))
    os.makedirs(output_dir, exist_ok=True)
    target_video_path = os.path.join(output_dir, os.path.basename(input_video))

    # --- 3. Interactive GUI Point Calibration ---
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    print("\n=== Select Calibration Frame ===")
    print("← : Previous frame")
    print("→ : Next frame")
    print("A : -30 frames")
    print("D : +30 frames")
    print("Enter : Use this frame")
    print("Esc : Cancel")

    while True:

        current_frame = max(0, min(current_frame, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        success, frame = cap.read()

        if not success:
            continue

        display = frame.copy()

        cv2.putText(
            display,
            f"Frame {current_frame}/{total_frames-1}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "LEFT/RIGHT: Step | A/D: +/-30 | ENTER: Select | ESC: Quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Select Calibration Frame", display)

        key = cv2.waitKeyEx(0)

        if key == 13:          # Enter
            first_frame = frame.copy()
            break

        elif key == 27:        # Esc
            cap.release()
            cv2.destroyAllWindows()
            return

        elif key == 2555904:   # Right Arrow
            current_frame += 1

        elif key == 2424832:   # Left Arrow
            current_frame -= 1

        elif key == ord('d') or key == ord('D'):
            current_frame += 30

        elif key == ord('a') or key == ord('A'):
            current_frame -= 30

    cap.release()
    cv2.destroyWindow("Select Calibration Frame")

    if not success:
        print(f"Error: Could not open video file at {input_video}")
        return

    calibration_frame = first_frame.copy()
    print("\n[INSTRUCTIONS] Click 4 points matching a single lane segment in order:")
    print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
    print("Press 'q' once all 4 points are placed to confirm and start tracking.")

    cv2.namedWindow("Click 4 Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    cv2.setMouseCallback("Click 4 Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left", 
                         click_event, param={'frame': calibration_frame})
    
    cv2.imshow("Click 4 Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left", calibration_frame)
    
    while len(clicked_points) < 4:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    if len(clicked_points) < 4:
        print("Error: Calibration canceled. 4 points were not selected.")
        cv2.destroyAllWindows()
        return

    cv2.destroyAllWindows()

    # Define source boundaries from user clicks and target flattened grid coordinates
    SOURCE = np.array(clicked_points, dtype=np.float32)
    
    preview = first_frame.copy()

    cv2.polylines(
        preview,
        [SOURCE.astype(np.int32)],
        True,
        (0,255,0),
        2
    )

    WIDTH_PIXELS = 400
    HEIGHT_PIXELS = 1200
    
    TARGET_PIXEL = np.array([
        [0, 0],
        [WIDTH_PIXELS, 0],
        [WIDTH_PIXELS, HEIGHT_PIXELS],
        [0, HEIGHT_PIXELS],
    ], dtype=np.float32)

    PIXELS_PER_METER_X = WIDTH_PIXELS / TARGET_WIDTH
    PIXELS_PER_METER_Y = HEIGHT_PIXELS / TARGET_HEIGHT

    view_transformer = ViewTransformer(
    source=SOURCE,
    target=TARGET_PIXEL
    )

    matrix = cv2.getPerspectiveTransform(
    SOURCE,
    TARGET_PIXEL
    )

    birdseye = cv2.warpPerspective(
        first_frame,
        matrix,
        (WIDTH_PIXELS, HEIGHT_PIXELS)
    )
    
    cv2.imshow("Original Selection", preview)
    cv2.imshow("Bird's Eye", birdseye)

    cv2.waitKey(0)

    cv2.destroyWindow("Original Selection")
    cv2.destroyWindow("Bird's Eye")


    # --- 4. Initialize Tracking Backend ---
    print("\nStarting Tracker Pipeline...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLO("yolo26n.pt").to(device)
    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.3)

    # Sizing annotators to look sharp at the native resolution
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_CENTER,
    )

    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

    # --- 5. Frame Processing Loop ---
    frame_generator = sv.get_video_frames_generator(source_path=input_video)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Run inference (Filtering for car, motorcycle, bus, truck)
            results = model(frame, conf=0.3, iou=0.7, classes=[2, 3, 5, 7])
            result = results[0]
            
            detections = sv.Detections.from_ultralytics(result)
            detections = byte_track.update_with_detections(detections=detections)

            names_dict = result.names

            # Transform contact coordinates to flat meter projection
            if len(detections) > 0:
                points = detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )

                transformed_points = view_transformer.transform_points(points=points)

                for tracker_id, (x_pix, y_pix) in zip(
                    detections.tracker_id,
                    transformed_points
                ):

                    x_meter = x_pix / PIXELS_PER_METER_X
                    y_meter = y_pix / PIXELS_PER_METER_Y

                    coordinates[tracker_id].append((x_meter, y_meter))

            # Generate formatted display labels
            labels = []
            for idx in range(len(detections)):
                if detections.tracker_id is None:
                    continue
                tracker_id = detections.tracker_id[idx]
                class_id = detections.class_id[idx]
                confidence = detections.confidence[idx]
                
                object_name = names_dict.get(class_id, "vehicle").upper()
                base_label = f"#{tracker_id} {object_name} ({confidence:.2f})"

                # Only evaluate speed if tracker has at least half a second of history
                if len(coordinates[tracker_id]) < int(video_info.fps / 2):
                    labels.append(base_label)
                else:
                    # Get distance components using 2D Pythagorean theorem across timeline
                    coord_start = coordinates[tracker_id][0]
                    coord_end = coordinates[tracker_id][-1]
                    
                    start = np.array(coordinates[tracker_id][0])
                    end = np.array(coordinates[tracker_id][-1])
                    distance_meters = abs(end[1] - start[1])
                    
                    time_seconds = len(coordinates[tracker_id]) / video_info.fps
                    
                    speed = (distance_meters / time_seconds) * speed_multiplier
                    labels.append(f"{base_label} | {int(speed)} {unit_label}")

            # Annotate and write to destination
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            sink.write_frame(annotated_frame)
            
            cv2.imshow("TrafficAI Vision Tracker", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    
    print(f"\nDone! Video compiled successfully.")
    print(f"Check output folder: {output_dir}")

if __name__ == "__main__":
    import torch  # Imported cleanly inside structural guard
    main()