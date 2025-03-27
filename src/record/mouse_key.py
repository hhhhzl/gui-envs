import time
from pynput import mouse, keyboard


class MouseKeyRecorder(object):
    def __init__(self, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.events = []
        self.stop_recording = False
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll
        )

    def start(self):
        print("Mouse and Keyboard Start Listening........")
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def _on_move(self, x, y):
        t = time.time() - self.start_time
        self.events.append((time.time(), t, 'MOVE', x, y))

    def _on_click(self, x, y, button, pressed):
        t = time.time() - self.start_time
        action = 'PRESSED' if pressed else 'RELEASED'
        self.events.append((time.time(), t, f'MOUSE_{action}', x, y, str(button)))

    def _on_scroll(self, x, y, dx, dy):
        t = time.time() - self.start_time
        self.events.append((time.time(), t, 'SCROLL', x, y, dx, dy))

    def _on_press(self, key):
        t = time.time() - self.start_time
        if key == keyboard.Key.esc:
            self.stop_recording = True
        try:
            self.events.append((time.time(), t, 'KEY_DOWN', key.char))
        except AttributeError:
            self.events.append((time.time(), t, 'KEY_DOWN', str(key)))

    def _on_release(self, key):
        t = time.time() - self.start_time
        try:
            self.events.append((time.time(), t, 'KEY_UP', key.char))
        except AttributeError:
            self.events.append((time.time(), t, 'KEY_UP', str(key)))
