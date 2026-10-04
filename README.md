# AI Gesture Mouse 🖱️🤖

A complete, high-performance, real-time **AI Gesture Mouse** application in Python that translates webcam hand tracking into native OS mouse control using OpenCV, MediaPipe, NumPy, and PyAutoGUI.

---

## 🌟 Key Features

- **⚡ Real-Time Responsiveness**: Target 30+ FPS pipeline with `pyautogui.PAUSE = 0.0` and safety failsafe protection.
- **📐 Virtual Bounding Box & Screen Mapping**: Configurable sub-frame touchpad area (`margin_x=100`, `margin_y=80`) mapped to full display resolution using `np.interp`.
- **🎯 EMA Smoothing Filter**: Exponential Moving Average (`smooth_factor = 5.0`) to eliminate micro-jitter and hand tremors.
- **🖐️ Deterministic State Machine**:
  - **Pointer Mode**: Track Index Tip (Landmark 8) with smooth cursor motion.
  - **Left Click**: Index (8) + Thumb (4) Pinch (`< 38px`, 0.35s debounce).
  - **Right Click**: Index (8) + Middle (12) Proximity (`< 30px`, 0.35s debounce).
  - **Double Click**: Index (8) + Pinky (20) Pinch (`< 38px`, 0.35s debounce).
  - **Drag & Drop**: Closed Fist detection held continuously for `>= 0.45s` (`mouseDown`), auto-released on open fist (`mouseUp`).
  - **Global Scroll**: Zone-based scrolling (Top 25% zone scroll up `+25`, Bottom 25% zone scroll down `-25`).
- **📺 Modern HUD Dashboard**: Visual landmark connections, neon virtual touchpad box, scroll zone overlays, dynamic mode status badge, FPS display, proximity indicator lines, and quick shortcut legend.

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.10+
- Webcam

### 2. Installation

Install required dependencies:

```bash
pip install opencv-python mediapipe numpy pyautogui
```

### 3. Execution

Launch the application with default settings:

```bash
python ai_gesture_mouse.py
```

Press **`q`** or **`ESC`** on the OpenCV camera window to exit cleanly.

---

## 🎮 Gesture Guide

| Gesture | Action | Condition | Visual Indicator |
| :--- | :--- | :--- | :--- |
| **Index Pointer** | Move Cursor | Move index finger tip inside virtual touchpad | Cyan tracked circle at index tip |
| **Index + Thumb Pinch** | Left Click | Index (8) to Thumb (4) distance `< 38px` | Green connection line & `PINCH!` badge |
| **Index + Middle Proximity**| Right Click | Index (8) to Middle (12) distance `< 30px` | Blue connection line & `PROXIMITY!` badge |
| **Index + Pinky Pinch** | Double Click | Index (8) to Pinky (20) distance `< 38px` | Magenta connection line & `PINCH!` badge |
| **Closed Fist (Hold >= 0.45s)**| Drag & Drop | All 4 fingertips curled below PIP joints | Orange hold progress bar -> Red `DRAGGING` |
| **Top 25% Zone** | Scroll Up | Index Tip in top 25% of frame (`y < cam_h * 0.25`)| Yellow top zone highlight & `SCROLL UP` |
| **Bottom 25% Zone** | Scroll Down | Index Tip in bottom 25% of frame (`y > cam_h * 0.75`)| Yellow bottom zone highlight & `SCROLL DOWN` |

---

## ⚙️ CLI Options & Customization

You can customize camera settings, margins, smoothing, and gesture thresholds via command-line arguments:

```bash
python ai_gesture_mouse.py --camera 0 --smooth-factor 5.0 --margin-x 100 --margin-y 80 --cooldown 0.35
```

### Options List:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--camera` | int | `0` | Camera device index |
| `--width` | int | `640` | Camera frame width |
| `--height` | int | `480` | Camera frame height |
| `--margin-x` | int | `100` | Horizontal margin for virtual touchpad |
| `--margin-y` | int | `80` | Vertical margin for virtual touchpad |
| `--smooth-factor` | float | `5.0` | EMA filter factor (higher = smoother, lower = faster response) |
| `--cooldown` | float | `0.35` | Click debounce cooldown duration in seconds |
| `--fist-hold-time` | float | `0.45` | Duration fist must be held to start dragging |
| `--left-click-thresh` | float | `38.0` | Pinch threshold in pixels for Left Click |
| `--right-click-thresh` | float | `30.0` | Proximity threshold in pixels for Right Click |
| `--double-click-thresh` | float | `38.0` | Pinch threshold in pixels for Double Click |

---

## 🧪 Running Unit Tests

Automated test suite covering EMA calculations, coordinate mapping, fist timer, and HUD visualizer:

```bash
python test_gesture_mouse.py
```

---

## 🛡️ Failsafe & Emergency Exit

- **PyAutoGUI Failsafe**: PyAutoGUI's built-in failsafe is set to `True`. Moving the mouse cursor into any corner of your screen will trigger the safety interlock and pause cursor events.
- **Window Exit**: Focus the camera window and press **`q`** or **`ESC`** to terminate. Any active `mouseDown` state will automatically release.
