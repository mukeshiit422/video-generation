"""
Creates a synthetic driving video for FOMM — a looping talking-head motion pattern
using keypoint interpolation. Saved to models/fomm/driving.mp4.
Run once: python utils/create_driving_video.py
"""
import numpy as np
import imageio
import cv2
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "models" / "fomm" / "driving.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H, FPS, DURATION = 256, 256, 25, 4
n_frames = FPS * DURATION

def draw_frame(t):
    img = np.full((H, W, 3), 200, dtype=np.uint8)
    # Slight head bob — very subtle vertical shift
    head_y = 128 + int(2 * np.sin(2 * np.pi * 0.5 * t))
    cv2.ellipse(img, (128, head_y), (70, 90), 0, 0, 360, (220, 190, 160), -1)
    # Eyes with slow blink every ~3s
    blink = 1 if (t % 3.0) > 0.15 else 0
    eye_h = 8 if blink else 2
    cv2.ellipse(img, (100, head_y - 18), (8, eye_h), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (156, head_y - 18), (8, eye_h), 0, 0, 360, (50, 50, 50), -1)
    # Smooth talking mouth — natural syllable rhythm at ~3 Hz, smooth sine
    mouth_open = int(9 * max(0, np.sin(2 * np.pi * 3.0 * t)) ** 0.6)
    cv2.ellipse(img, (128, head_y + 32), (22, max(2, mouth_open)), 0, 0, 360, (100, 60, 60), -1)
    return img

frames = [draw_frame(i / FPS) for i in range(n_frames)]
imageio.mimwrite(str(OUT), frames, fps=FPS, codec="libx264")
print(f"Driving video saved → {OUT}  ({n_frames} frames @ {FPS}fps)")
