import cv2
import torch
import numpy as np
from ultralytics import YOLO

# YOLOモデル読み込み
model = YOLO("yolov8x.pt")

# 画像を指定して推論
results = model.predict("ex3.jpg", conf=0.1)

# 元画像取得
img = results[0].orig_img

# バウンディングボックス取得
boxes = results[0].boxes

# HSV色範囲（赤青黄）
lower_red = np.array([160, 50, 50])
upper_red = np.array([180, 255, 255])
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([150, 255, 255])
lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([45, 255, 255])

# 色判定関数（最大連結成分の面積を返す）
def get_max_area(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return 0
    return max(stats[1:, cv2.CC_STAT_AREA])

# 例：コート範囲（x_min, y_min, x_max, y_max）を仮に指定
court_x1, court_y1 = 100, 100
court_x2, court_y2 = 4000, 4000  # ex3.jpg の解像度に合わせて調整してください

def is_inside_court(x, y):
    return court_x1 <= x <= court_x2 and court_y1 <= y <= court_y2

for box in boxes:
    cls_id = int(box.data[0][5])
    if cls_id != 0:
        continue

    x1, y1, x2, y2 = map(int, box.data[0][0:4])
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2  # 中心座標

    # コート内でないならスキップ（観客など）
    if not is_inside_court(cx, cy):
        continue

    # ROI 抽出と色判定処理はそのまま
    cx1 = x1 + int(w * 0.35)
    cx2 = x1 + int(w * 0.65)
    cy1 = y1 + int(h * 0.2)
    cy2 = y1 + int(h * 0.4)
    roi = img[cy1:cy2, cx1:cx2]

    if roi.size == 0:
        continue

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_roi, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    red_area = get_max_area(red_mask)
    blue_area = get_max_area(blue_mask)
    yellow_area = get_max_area(yellow_mask)

    if yellow_area > 80:
        color = (0, 255, 255)
    elif red_area > 1 or blue_area > 1:
        color = (0, 0, 255)
    else:
        continue

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)



# 表示・保存
cv2.imshow("ユニフォーム分類", img)
cv2.imwrite("out_classified.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
