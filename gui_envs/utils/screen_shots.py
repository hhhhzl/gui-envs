import cv2
import mss
import pyautogui
import numpy as np
import os
import time

sct = mss.mss()
monitor = sct.monitors[0]


def take_screenshot():
    img = np.array(sct.grab(monitor))
    # img = np.array(pyautogui.screenshot(region=[0, 0, 1440, 900]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img[:, :, :3]
    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite(f"screenshots/screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png", img)
    return img


if __name__ == "__main__":
    while True:
        take_screenshot()
        time.sleep(5)
