import cv2
import os
import time
import datetime
from ultralytics import YOLO

save_dir = 'media'
save_dir_vid = 'video'

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir_vid, exist_ok=True)

cap = cv2.VideoCapture('cat_video1.mp4')

classes = ['person', 'airplane', 'cat', 'car', 'dog']
if not cap.isOpened():
    print("Camera not found")
    exit()


model = YOLO("yolov8n.pt")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
date_name = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
video_name = f'{save_dir_vid}/video_{date_name}.mp4'
# out = cv2.VideoWriter(video_name, fourcc, frame_fps, (frame_width, frame_height))

image_name = f'{save_dir}/photo_{date_name}.jpg'

fps = 0

while True:
    fps_start = time.time()
    ret, frame = cap.read()

    if not ret:
        print("No frame")

    result = model(frame, stream=True, conf=0.5)

    for i in result:
        for n in i.boxes:
            cls = int(n.cls[0])
            label = model.names[cls]
            conf = round(float(n.conf[0]), 2)

            if label in classes:

                x, y, w, h = map(int, n.xyxy[0])
                cv2.rectangle(frame, (x,y), (w,h), (0,255,0), 2)

                cv2.putText(frame, f'{label}, {conf * 100}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


    cv2.putText(frame, 'AI 5',(300,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (120, 225, 252), 3)
    cv2.putText(frame, date_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (79, 201, 55), 2)

    fps_end = time.time()

    fps = 1 / (fps_end - fps_start)
    cv2.putText(frame, f'FPS: {round(fps, 1)}', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite(image_name, frame)
        cv2.putText(frame, 'Saved', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
        cv2.imshow('S for take photo',frame)
        cv2.waitKey(3)
    if key == ord('v'):
        out = cv2.VideoWriter(video_name, fourcc, frame_fps, (frame_width, frame_height))

cap.release()
cv2.destroyAllWindows()

