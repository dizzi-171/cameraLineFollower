import cv2
import numpy as np

def expand_box(box, scale=1.6):
    center = np.mean(box, axis=0)
    expanded = (box - center) * scale + center
    return np.intp(expanded)

def order_box_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect.astype(int)

def draw_quadrant_lines_and_labels(frame, box):
    expanded_box = expand_box(box, scale=1.6)
    rect = order_box_points(expanded_box)

    top_mid = ((rect[0] + rect[1]) // 2)
    bottom_mid = ((rect[2] + rect[3]) // 2)
    left_mid = ((rect[0] + rect[3]) // 2)
    right_mid = ((rect[1] + rect[2]) // 2)

    cv2.line(frame, tuple(left_mid), tuple(right_mid), (255, 255, 255), 1)
    cv2.line(frame, tuple(top_mid), tuple(bottom_mid), (255, 255, 255), 1)
    cv2.polylines(frame, [expanded_box], isClosed=True, color=(255, 255, 255), thickness=1)

    offset = 10
    labels = {
        "(-x, +y)": rect[0] - [offset, offset],
        "(+x, +y)": rect[1] + [offset, -offset],
        "(+x, -y)": rect[2] + [offset, offset],
        "(-x, -y)": rect[3] - [offset, -offset],
    }

    for text, pos in labels.items():
        x, y = pos.astype(int)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    return rect

def detectar_quadrantes(frame):
    frame = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 60, 60])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_threshold = 100

    left = 0
    right = 0

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        ordered_box = order_box_points(box)
        expanded_box = expand_box(ordered_box, scale=1.8)

        mask_expanded = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_expanded, [expanded_box], 255)

        mask_inner = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_inner, [ordered_box], 255)

        mask_ring = cv2.bitwise_and(mask_expanded, cv2.bitwise_not(mask_inner))

        cx = int(np.mean([p[0] for p in expanded_box]))
        cy = int(np.mean([p[1] for p in expanded_box]))

        quadrants = {
            "(-x, +y)": np.zeros_like(mask_ring),
            "(+x, +y)": np.zeros_like(mask_ring),
            "(+x, -y)": np.zeros_like(mask_ring),
            "(-x, -y)": np.zeros_like(mask_ring),
        }

        for y in range(mask_ring.shape[0]):
            for x in range(mask_ring.shape[1]):
                if mask_ring[y, x]:
                    if x < cx and y < cy:
                        quadrants["(-x, +y)"][y, x] = 255
                    elif x >= cx and y < cy:
                        quadrants["(+x, +y)"][y, x] = 255
                    elif x >= cx and y >= cy:
                        quadrants["(+x, -y)"][y, x] = 255
                    elif x < cx and y >= cy:
                        quadrants["(-x, -y)"][y, x] = 255

        dark_counts = {}
        for q, mask in quadrants.items():
            total_pixels = cv2.countNonZero(mask)
            if total_pixels == 0:
                percentage = 0
            else:
                dark_pixels = cv2.countNonZero(cv2.bitwise_and((gray < dark_threshold).astype(np.uint8) * 255, mask))
                percentage = dark_pixels / total_pixels
            dark_counts[q] = percentage

        chosen_quadrant = max(dark_counts, key=dark_counts.get)

        
        # Definindo mensagem a exibir
        msg = ""
        if chosen_quadrant.endswith("-y)"):
            msg = "False green"
        else:  # +y
            if chosen_quadrant.startswith("(-x"):
                msg = (f"right {chosen_quadrant}")
                right = chosen_quadrant
            else:
                msg = (f"left{chosen_quadrant}")
                left = chosen_quadrant
 
            

        cv2.drawContours(frame, [ordered_box], 0, (0, 0, 255), 2)
        draw_quadrant_lines_and_labels(frame, box)
        cx_text = int(np.mean(box[:, 0]))
        cy_text = int(np.mean(box[:, 1]))
        cv2.putText(frame, msg, (cx_text - 30, cy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    if left and right:
        msg = (f"Dois verdes: E{left} D{right}")
        cv2.putText(frame,msg, (0,20), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0), 1, cv2.LINE_AA)
    return frame

# Captura da câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultado = detectar_quadrantes(frame)
    cv2.imshow("Quadrantes com múltiplos verdes", resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
