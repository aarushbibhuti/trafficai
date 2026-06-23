import os
from ultralytics import YOLO
from datetime import datetime


def main():

    model = YOLO("yolo26n.pt")

    input_video = "data/videos/trafficshort.mp4"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    abs_project_root = os.path.abspath("data/processed")

    print("starting tracker...")
    print(f"files wil be saved to: {abs_project_root}\n")

    model.track(
        source=input_video,
        persist=True,
        tracker="bytetrack.yaml",
        save=True,
        project=abs_project_root,
        name=f"track_{timestamp}",
        classes= [2, 3, 5, 7],
    )

    print(f"Done. Check {abs_project_root}")

if __name__ == "__main__":
    main()