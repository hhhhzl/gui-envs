import cv2
import numpy as np
import mss
import time
from src.recording.mouse_key import MouseKeyRecorder


class ScreenRecorder(object):
    def __init__(
            self,
            filename,
            fps=20,
            monitor=0,
            start_time=None,
            mkr: MouseKeyRecorder = None
    ):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]
        self.monitor_index = monitor
        self.screen_size = self.monitor
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        actual_height, actual_width = frame.shape[:2]
        self.resolution = (actual_width, actual_height)
        self.filename = filename
        self.fps = fps
        self.time_offset = time.time() - self.start_time
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.filename, fourcc, fps, self.resolution)
        self.frames = 0
        self.duration = 0

        if mkr is None:
            self.mkr = MouseKeyRecorder()
            self.mkr.keyboard_listener.start()
        else:
            self.mkr = mkr

    def start(self):
        if self.mkr is None:
            print("Control + C to stop Recording")
            try:
                print("Screen Recording Start........")
                print("Press ESC to stop Recording")
                self.time_offset = time.time() - self.start_time
                while True:
                    screenshot = self.sct.grab(self.monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.out.write(frame)
            except KeyboardInterrupt:
                self.out.release()
                cv2.destroyAllWindows()
        else:
            print("Screen Recording Start........")
            print("Press ESC to stop Recording")
            self.time_offset = time.time() - self.start_time
            while not self.mkr.stop_recording:
                screenshot = self.sct.grab(self.monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.out.write(frame)

            self.out.release()
            cv2.destroyAllWindows()
            print("Recorded Screen Saved.")

    def write_record(self):
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            return
        else:
            self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps > 0:
                self.duration = self.frames / self.fps
        cap.release()