import cv2
import numpy as np
import mss


def record_screen(filename, fps=20):
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

        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    record_screen("../utils/test.mp4", 20)
