from screen import ScreenRecorder
from mouse_key import MouseKeyRecorder
import time


def record_screen():
    ScreenRecorder(filename='test.mp4').start()


def record_keyboard():
    MouseKeyRecorder().start()


def record():
    start_time = time.time()

    # define keyboard and mouse
    mc = MouseKeyRecorder(start_time=start_time)
    sc = ScreenRecorder(filename='test.mp4', start_time=start_time, mkr=mc)
    mc.start()
    sc.start()

    print("\nRecording stopped. Check 'events_log.txt'.")
    with open('events_log.txt', 'w') as f:
        for ev in mc.events:
            f.write(str(ev) + "\n")
    f.close()
    print("Done. Events saved to events_log.txt")


if __name__ == "__main__":
    record()
