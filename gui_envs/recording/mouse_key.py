import time
from pynput import mouse, keyboard


class MouseKeyRecorder(object):
    def __init__(self, start_time=None):
        super().__init__()
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.events = []
        self.stop_recording = False
        self.keyboard_listener = keyboard.Listener(
            on_press=self.__on_press,
            on_release=self.__on_release
        )
        self.mouse_listener = mouse.Listener(
            on_move=self.__on_move,
            on_click=self.__on_click,
            on_scroll=self.__on_scroll
        )
        self.ending_time = None

    def start(self, ):
        print("Mouse and Keyboard Start Listening........")
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def write_record(self, path, file_name):
        try:
            with open(f"{path}/{file_name}.txt", 'w') as f:
                for ev in self.events:
                    f.write(str(ev) + "\n")
            f.close()
            print("Keyboards Actions Saved")
        except Exception as e:
            print(f"Keyboards Actions Save Failed: {str(e)}")

    def __on_move(self, x, y):
        t = time.time() - self.start_time
        self.events.append([time.time(), 'MOVE', x, y])

    def __on_click(self, x, y, button, pressed):
        t = time.time() - self.start_time
        action = 'PRESSED' if pressed else 'RELEASED'
        self.events.append([time.time(), f'MOUSE_{action}', x, y, str(button)])

    def __on_scroll(self, x, y, dx, dy):
        t = time.time() - self.start_time
        self.events.append([time.time(), 'SCROLL', x, y, dx, dy])

    def __on_press(self, key):
        t = time.time() - self.start_time
        if key == keyboard.Key.esc:
            self.ending_time = time.time()
            self.stop_recording = True
        try:
            self.events.append([time.time(), 'KEY_DOWN', key.char])
        except AttributeError:
            self.events.append([time.time(), 'KEY_DOWN', str(key)])

    def __on_release(self, key):
        t = time.time() - self.start_time
        try:
            self.events.append([time.time(), 'KEY_UP', key.char])
        except AttributeError:
            self.events.append([time.time(), 'KEY_UP', str(key)])
