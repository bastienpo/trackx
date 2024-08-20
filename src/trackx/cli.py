import cv2
import numpy as np
from PIL import Image, ImageDraw
from detector import Detector

if __name__ == "__main__":
    detector = Detector()

    cap = cv2.VideoCapture("data/ADL-Rundle-6-raw.webm", cv2.CAP_FFMPEG)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 4 != 0:
            frame_number += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        boxes = detector.detect(frame, "person")
        for box in boxes:
            draw = ImageDraw.Draw(frame)
            draw.rectangle(box, outline="red", width=2)

        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
