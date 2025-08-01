import cv2
import time
import numpy as np
from ultralytics import solutions

# CONFIG
video_path = "traffic2.mp4"
processing_size = (480, 270)
region_points = [[437, 452], [1675, 441], [1896, 983], [30, 940]]
skip_frame = 2  # bỏ qua khung để tăng FPS

# VIDEO SETUP
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Can't open video: {video_path}"

ret, im0 = cap.read()
assert ret, "Can't read first frame"
original_size = im0.shape[1], im0.shape[0]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def scale_region(region, orig_size, new_size):
    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]
    return [[int(x * scale_x), int(y * scale_y)] for x, y in region]

region_scaled = scale_region(region_points, original_size, processing_size)

# INIT COUNTER
counter = solutions.ObjectCounter(
    model="yolov8n.pt",
    region=region_scaled,
    classes=[2, 5, 7],  # Car, Bus, Truck
    show=False,          # Tắt vẽ tự động
    verbose=False,
    show_labels=False,
    show_conf=False
)

# GPU nếu có
if counter.model.device.type != "cuda":
    try:
        counter.model.to("cuda")
        print("Switched to GPU")
    except:
        print("No GPU found, using CPU")

cv2.namedWindow("Counting Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Counting Result", 960, 540)

frame_id = 0
while cap.isOpened():
    start = time.time()
    success, im0 = cap.read()
    if not success:
        break

    frame_id += 1
    if frame_id % skip_frame != 0:
        continue

    resized = cv2.resize(im0, processing_size)

    blank = np.zeros_like(resized)  # ảnh đen, không cần hiển thị
    counter.im0 = blank             # để ObjectCounter không có gì để vẽ vào
    counter(resized)

    overlay = resized.copy()

    # Vẽ vùng đếm
    cv2.polylines(overlay, [np.array(region_scaled, dtype=np.int32)],
                  isClosed=True, color=(0, 255, 255), thickness=2)

    # Vẽ nhãn từng object
    for i in range(len(counter.boxes)):
        x1, y1, x2, y2 = map(int, counter.boxes[i])
        cls_id = int(counter.clss[i])
        track_id = int(counter.track_ids[i])
        label = f"{counter.model.names[cls_id]} {track_id}"
        cv2.putText(overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # === Resize lại về gốc để hiển thị rõ ===
    overlay = cv2.resize(overlay, original_size)

    # Show lên màn hình
    cv2.imshow("Counting Result", overlay)
    print(f"FPS: {1 / (time.time() - start):.2f}")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()