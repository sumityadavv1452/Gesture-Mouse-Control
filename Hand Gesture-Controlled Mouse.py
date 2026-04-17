import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Setup ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
# Using 0.7 confidence for high stability
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = True

# --- Configuration & States ---
ZONE_PCT = 0.25      # Top/Bottom 25% for scrolling
SMOOTHING = 8        # Higher = less jitter
prev_x, prev_y = 0, 0
pinch_count = 0
last_pinch_time = 0
fist_start = None
is_dragging = False

def get_dist(p1, p2, w, h):
    return np.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # HUD: Drawing Scroll Zones
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, int(h*ZONE_PCT)), (0, 0, 150), -1) # Red Top
    cv2.rectangle(overlay, (0, int(h*(1-ZONE_PCT))), (w, h), (150, 0, 0), -1) # Blue Bottom
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0].landmark
        
        # Mapping Landmarks
        t, i, m, r, p = lms[4], lms[8], lms[12], lms[16], lms[20] # Tips
        i_k, m_k, r_k, p_k = lms[6], lms[10], lms[14], lms[18]   # Knuckles

        # 1. MOVEMENT (Index Only)
        # Using 60% central mapping (0.2 to 0.8 range)
        mx = np.interp(i.x, [0.2, 0.8], [0, screen_w])
        my = np.interp(i.y, [0.2, 0.8], [0, screen_h])
        
        curr_x = prev_x + (mx - prev_x) / SMOOTHING
        curr_y = prev_y + (my - prev_y) / SMOOTHING
        
        if not is_dragging:
            pyautogui.moveTo(curr_x, curr_y, _pause=False)
        else:
            pyautogui.dragTo(curr_x, curr_y, _pause=False)
        prev_x, prev_y = curr_x, curr_y

        # 2. LEFT CLICK (Index + Thumb Pinch)
        if get_dist(i, t, w, h) < 40:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (50, 50), 1, 2, (0, 255, 0), 2)
            time.sleep(0.2)

        # 3. RIGHT CLICK (Index + Middle Pinch)
        if get_dist(i, m, w, h) < 40 and get_dist(i, t, w, h) > 60:
            pyautogui.rightClick()
            cv2.putText(frame, "RIGHT CLICK", (50, 50), 1, 2, (0, 255, 255), 2)
            time.sleep(0.4)

        # 4. DOUBLE CLICK (Index + Pinky Pinch x2 in 2s)
        if get_dist(i, p, w, h) < 50:
            curr_t = time.time()
            if curr_t - last_pinch_time > 0.3:
                pinch_count += 1
                last_pinch_time = curr_t
            if pinch_count == 2:
                pyautogui.doubleClick()
                pinch_count = 0
                time.sleep(0.4)
        elif time.time() - last_pinch_time > 2.0:
            pinch_count = 0

        # 5. SCROLL (Index + Middle + Ring UP)
        if i.y < i_k.y and m.y < m_k.y and r.y < r_k.y:
            if i.y < ZONE_PCT: pyautogui.scroll(40)
            elif i.y > (1 - ZONE_PCT): pyautogui.scroll(-40)

        # 6. DRAG (Fist Hold 0.45s)
        fist = all(lms[tip].y > lms[kn].y for tip, kn in [(8,6), (12,10), (16,14), (20,18)])
        if fist:
            if fist_start is None: fist_start = time.time()
            if time.time() - fist_start > 0.45:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                cv2.putText(frame, "DRAG ON", (w-150, 50), 1, 1, (0,0,255), 2)
        else:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            fist_start = None

    cv2.imshow("Sumit's Final Gesture Mouse", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()