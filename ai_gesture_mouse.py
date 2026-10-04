"""
AI Gesture Mouse - Real-time Hand Tracking Cursor Controller
============================================================
Translates webcam hand gestures into native OS mouse events using OpenCV, MediaPipe Tasks API, NumPy, and PyAutoGUI.
Compatible with Python 3.10+ and Python 3.14+.
"""

import cv2
import os
import time
import math
import argparse
import urllib.request
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Standard Hand Landmark Skeleton Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"


def ensure_model(model_path: str = MODEL_PATH) -> str:
    """Ensures MediaPipe model file exists locally, downloading automatically if missing."""
    if not os.path.exists(model_path):
        print("[INFO] Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class SmoothingFilter:
    """Exponential Moving Average (EMA) filter to eliminate hand tremors."""

    def __init__(self, smooth_factor: float = 5.0):
        self.smooth_factor = max(1.0, smooth_factor)
        self.prev_x = None
        self.prev_y = None

    def update(self, target_x: float, target_y: float):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = target_x, target_y
            return target_x, target_y

        curr_x = self.prev_x + (target_x - self.prev_x) / self.smooth_factor
        curr_y = self.prev_y + (target_y - self.prev_y) / self.smooth_factor
        self.prev_x, self.prev_y = curr_x, curr_y
        return curr_x, curr_y

    def reset(self):
        self.prev_x = None
        self.prev_y = None


class HandTracker:
    """Encapsulates MediaPipe Tasks API HandLandmarker detection and calculations."""

    def __init__(self, model_path: str = MODEL_PATH):
        actual_model = ensure_model(model_path)
        base_options = python.BaseOptions(model_asset_path=actual_model)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray):
        """Processes BGR frame and returns landmark pixel coordinates list or None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.landmarker.detect(mp_image)

        if not res or not res.hand_landmarks or len(res.hand_landmarks) == 0:
            return None

        h, w, _ = frame_bgr.shape
        return [(int(lm.x * w), int(lm.y * h)) for lm in res.hand_landmarks[0]]

    @staticmethod
    def draw_skeleton(frame: np.ndarray, lm_list: list):
        """Draws pure OpenCV skeleton lines and joint dots on frame."""
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, lm_list[p1], lm_list[p2], (0, 255, 200), 2, cv2.LINE_AA)

        for idx, pt in enumerate(lm_list):
            if idx in (4, 8, 12, 16, 20):  # Fingertips
                cv2.circle(frame, pt, 6, (255, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (0, 255, 100), -1, cv2.LINE_AA)

    @staticmethod
    def distance(p1: tuple, p2: tuple) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def is_fist(lm: list) -> bool:
        """Detects closed fist state (fingertips 8, 12, 16, 20 curled below PIPs 6, 10, 14, 18)."""
        return lm[8][1] > lm[6][1] and lm[12][1] > lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]

    # Aliases
    calculate_distance = distance
    draw_custom_landmarks = draw_skeleton


class MouseController:
    """Translates gestures to native OS mouse actions via PyAutoGUI."""

    def __init__(self, cam_w=640, cam_h=480, margin_x=100, margin_y=80, smooth_factor=5.0, debounce_cooldown=0.35, fist_hold_time=0.45):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.screen_w, self.screen_h = pyautogui.size()

        self.filter = SmoothingFilter(smooth_factor=smooth_factor)
        self.cooldown = debounce_cooldown
        self.fist_hold_time = fist_hold_time

        self.last_click_time = 0.0
        self.last_scroll_time = 0.0
        self.fist_start_time = None
        self.is_dragging = False

        pyautogui.PAUSE = 0.0
        pyautogui.FAILSAFE = True

    def map_to_screen(self, x: float, y: float):
        """Maps virtual touchpad coordinates to screen bounds with EMA smoothing."""
        tx = np.clip(np.interp(x, [self.margin_x, self.cam_w - self.margin_x], [0, self.screen_w]), 0, self.screen_w - 1)
        ty = np.clip(np.interp(y, [self.margin_y, self.cam_h - self.margin_y], [0, self.screen_h]), 0, self.screen_h - 1)
        sx, sy = self.filter.update(tx, ty)
        return int(sx), int(sy)

    def move(self, sx: int, sy: int):
        try:
            pyautogui.moveTo(sx, sy)
        except pyautogui.FailSafeException:
            pass

    def click(self, button='left', double=False) -> bool:
        now = time.time()
        if now - self.last_click_time >= self.cooldown:
            try:
                if double:
                    pyautogui.doubleClick()
                else:
                    pyautogui.click(button=button)
            except pyautogui.FailSafeException:
                pass
            self.last_click_time = now
            return True
        return False

    def update_drag(self, is_fist: bool = False, is_fist_active: bool = None):
        if is_fist_active is not None:
            is_fist = is_fist_active

        now = time.time()
        progress = 0.0

        if is_fist:
            if self.fist_start_time is None:
                self.fist_start_time = now
            progress = now - self.fist_start_time
            if progress >= self.fist_hold_time and not self.is_dragging:
                try:
                    pyautogui.mouseDown(button='left')
                except pyautogui.FailSafeException:
                    pass
                self.is_dragging = True
        else:
            if self.is_dragging:
                try:
                    pyautogui.mouseUp(button='left')
                except pyautogui.FailSafeException:
                    pass
                self.is_dragging = False
            self.fist_start_time = None

        return self.is_dragging, progress

    def scroll(self, amount: int) -> bool:
        now = time.time()
        if now - self.last_scroll_time >= 0.05:
            try:
                pyautogui.scroll(amount)
            except pyautogui.FailSafeException:
                pass
            self.last_scroll_time = now
            return True
        return False

    def release_drag(self):
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button='left')
            except pyautogui.FailSafeException:
                pass
            self.is_dragging = False

    # Alias
    update_drag_state = update_drag


class HUDVisualizer:
    """Renders visual feedback overlay, status dashboard, and guide lines."""

    def __init__(self, cam_w=640, cam_h=480, margin_x=100, margin_y=80):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.scroll_top = int(cam_h * 0.25)
        self.scroll_bottom = int(cam_h * 0.75)

    def draw(
        self,
        frame: np.ndarray,
        mode: str = "TRACKING",
        fps: float = 30.0,
        lm: list = None,
        dists: dict = None,
        fist_prog: float = 0.0,
        mode_text: str = None,
        fist_hold_progress: float = None,
        gesture_distances: dict = None,
        lm_list: list = None
    ):
        if mode_text is not None:
            mode = mode_text
        if fist_hold_progress is not None:
            fist_prog = fist_hold_progress
        if gesture_distances is not None:
            dists = gesture_distances
        if lm_list is not None:
            lm = lm_list

        # 1. Scroll Zones Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.cam_w, self.scroll_top), (40, 40, 100), -1)
        cv2.rectangle(overlay, (0, self.scroll_bottom), (self.cam_w, self.cam_h), (40, 40, 100), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        cv2.line(frame, (0, self.scroll_top), (self.cam_w, self.scroll_top), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (0, self.scroll_bottom), (self.cam_w, self.scroll_bottom), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "▲ SCROLL UP", (15, self.scroll_top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        cv2.putText(frame, "▼ SCROLL DOWN", (15, self.scroll_bottom + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        # 2. Virtual Bounding Box
        x1, y1 = self.margin_x, self.margin_y
        x2, y2 = self.cam_w - self.margin_x, self.cam_h - self.margin_y
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, "VIRTUAL TOUCHPAD", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)

        # 3. Gesture Interacting Lines & Highlights
        if lm and dists:
            idx, thumb, mid, pinky = lm[8], lm[4], lm[12], lm[20]

            dist_left = dists.get("left", dists.get("left_click", 999))
            dist_right = dists.get("right", dists.get("right_click", 999))
            dist_double = dists.get("double", dists.get("double_click", 999))

            if dist_left < 38:
                cv2.line(frame, idx, thumb, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "PINCH!", (idx[0] + 10, idx[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            if dist_right < 30:
                cv2.line(frame, idx, mid, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "PROXIMITY!", (idx[0] + 10, idx[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
            if dist_double < 38:
                cv2.line(frame, idx, pinky, (255, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "PINCH!", (idx[0] + 10, idx[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

            if 0.0 < fist_prog < 0.45:
                pct = min(1.0, fist_prog / 0.45)
                bx, by = lm[5][0] - 50, lm[5][1] - 25
                cv2.rectangle(frame, (bx, by), (bx + 100, by + 8), (50, 50, 50), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(100 * pct), by + 8), (0, 165, 255), -1)

            cv2.circle(frame, idx, 7, (0, 255, 255), -1, cv2.LINE_AA)

        # 4. Top Dashboard Header
        hdr = frame.copy()
        cv2.rectangle(hdr, (0, 0), (self.cam_w, 36), (15, 15, 20), -1)
        cv2.addWeighted(hdr, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "AI GESTURE MOUSE", (120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        mode_str = f"MODE: {mode}"
        (tw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (self.cam_w - tw - 20, 6), (self.cam_w - 8, 30), (0, 255, 200), -1)
        cv2.putText(frame, mode_str, (self.cam_w - tw - 14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # 5. Bottom Legend
        cv2.putText(frame, "Index: Move | Idx+Thumb: L-Click | Idx+Mid: R-Click | Idx+Pinky: Dbl-Click | Fist: Drag | 'q': Quit",
                    (10, self.cam_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        return frame

    # Alias
    draw_hud = draw


class GestureMouseApp:
    """Main application loop managing camera capture, gesture machine, and mouse control."""

    def __init__(self, camera_idx=0, width=640, height=480, margin_x=100, margin_y=80, smooth_factor=5.0):
        self.cam_w = width
        self.cam_h = height
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_idx}")

        self.tracker = HandTracker()
        self.mouse = MouseController(self.cam_w, self.cam_h, margin_x, margin_y, smooth_factor)
        self.hud = HUDVisualizer(self.cam_w, self.cam_h, margin_x, margin_y)

        self.prev_time = time.time()

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                lm_list = self.tracker.process(frame)

                mode = "NO HAND"
                dists = {}
                fist_prog = 0.0

                if lm_list:
                    HandTracker.draw_skeleton(frame, lm_list)
                    mode = "TRACKING"

                    idx, thumb, mid, pinky, idx_mcp = lm_list[8], lm_list[4], lm_list[12], lm_list[20], lm_list[5]

                    dist_left = HandTracker.distance(idx, thumb)
                    dist_right = HandTracker.distance(idx, mid)
                    dist_dbl = HandTracker.distance(idx, pinky)
                    dists = {"left": dist_left, "right": dist_right, "double": dist_dbl}

                    is_fist = HandTracker.is_fist(lm_list)
                    is_dragging, fist_prog = self.mouse.update_drag(is_fist)

                    if is_dragging:
                        mode = "DRAGGING"
                        sx, sy = self.mouse.map_to_screen(idx_mcp[0], idx_mcp[1])
                        self.mouse.move(sx, sy)
                    elif is_fist:
                        mode = "HOLDING FIST..."
                    else:
                        sx, sy = self.mouse.map_to_screen(idx[0], idx[1])
                        self.mouse.move(sx, sy)

                        if dist_left < 38:
                            if self.mouse.click(button='left'):
                                mode = "LEFT CLICK"
                        elif dist_right < 30:
                            if self.mouse.click(button='right'):
                                mode = "RIGHT CLICK"
                        elif dist_dbl < 38:
                            if self.mouse.click(double=True):
                                mode = "DOUBLE CLICK"
                        else:
                            if idx[1] < self.cam_h * 0.25:
                                if self.mouse.scroll(25):
                                    mode = "SCROLL UP"
                            elif idx[1] > self.cam_h * 0.75:
                                if self.mouse.scroll(-25):
                                    mode = "SCROLL DOWN"

                now = time.time()
                fps = 1.0 / (now - self.prev_time) if (now - self.prev_time) > 0 else 30.0
                self.prev_time = now

                frame = self.hud.draw(frame, mode, fps, lm_list, dists, fist_prog)
                cv2.imshow("AI Gesture Mouse", frame)

                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
        finally:
            self.mouse.release_drag()
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gesture Mouse")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--smooth-factor", type=float, default=5.0, help="EMA smoothing factor")
    parser.add_argument("--margin-x", type=int, default=100, help="Touchpad margin X")
    parser.add_argument("--margin-y", type=int, default=80, help="Touchpad margin Y")
    args = parser.parse_args()

    app = GestureMouseApp(
        camera_idx=args.camera,
        smooth_factor=args.smooth_factor,
        margin_x=args.margin_x,
        margin_y=args.margin_y
    )
    app.run()
