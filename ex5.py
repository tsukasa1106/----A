import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 画像読み込み（ローカルファイル用）
img = cv2.imread("ex2.jpg")

# モデル読み込み
model = YOLO("yolov8x.pt")

# 推論（人物検出）
results = model.predict(img, conf=0.3)
boxes = results[0].boxes
cls_ids = results[0].boxes.cls.cpu().numpy()
xyxy = boxes.xyxy.cpu().numpy()

# HSVで青色判定する範囲（Hue: 90〜130程度）
lower_blue = np.array([90, 80, 40])
upper_blue = np.array([140, 255, 255])

# 人物ごとに判定
for box, cls_id in zip(xyxy, cls_ids):
    if int(cls_id) != 0:  # person以外はスキップ
        continue

    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
    blue_ratio = np.sum(mask > 0) / mask.size

    if blue_ratio > 0.03:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

# 結果表示
cv2.imshow("青い服の人物", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
