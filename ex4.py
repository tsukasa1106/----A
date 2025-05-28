from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolov8n.pt")  # 軽量モデルに変更

results = model.predict("ex2.jpg", conf=0.1, device="cpu")  # CPUで処理

img = results[0].orig_img
boxes = results[0].boxes

# 面積最大のboxを探す
max_area = 0
max_box = None

for box in boxes:
    x1, y1, x2, y2 = box.data[0][0:4]
    area = (x2 - x1) * (y2 - y1)
    if area > max_area:
        max_area = area
        max_box = (int(x1), int(y1), int(x2), int(y2))

# 最大面積のboxだけを描画
if max_box:
    cv2.rectangle(img, max_box[:2], max_box[2:], (0, 0, 255), thickness=3)

cv2.imshow("最大物体領域", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
