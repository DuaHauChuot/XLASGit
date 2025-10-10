import cv2
import numpy as np

# Thay đường dẫn video input/output
INPUT_VIDEO = "task1.mp4"
OUTPUT_VIDEO = "task1output.mp4"

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxes = []

    # Ví dụ: phát hiện vùng màu đỏ (nhiều biển báo có viền đỏ)
    # Khoảng 1: đỏ thấp
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    # Khoảng 2: đỏ cao
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # optional: blur + morphological to reduce noise
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # lọc nhiễu nhỏ, điều chỉnh tuỳ video
            continue
        x,y,w,h = cv2.boundingRect(cnt)

        # Tùy chọn: kiểm tra tỉ lệ hoặc shape (tam giác/chữ nhật/...), ví dụ:
        aspect = w / float(h)
        if 0.5 < aspect < 1.6:  # loại bỏ những vùng quá hẹp
            boxes.append((x,y,w,h))

    # Vẽ bounding boxes
    for (x,y,w,h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Sign", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Cannot open video:", INPUT_VIDEO)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out.write(frame)

        # optional: show live
        cv2.imshow("Detect Signs", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
