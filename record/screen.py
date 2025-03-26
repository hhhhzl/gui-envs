import cv2
import numpy as np
import mss
from pynput import keyboard

stop_recording = False


def on_press(key):
    global stop_recording
    if key == keyboard.Key.esc:
        stop_recording = True
        return False


def record_screen(filename, fps=20):
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Actual pixel dimensions
        actual_height, actual_width = frame.shape[:2]
        resolution = (actual_width, actual_height)
        print(resolution)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps, resolution)

        while not stop_recording:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        listener.stop()


if __name__ == "__main__":
    record_screen("test.mp4", 20)
