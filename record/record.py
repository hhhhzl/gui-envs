from pynput import mouse, keyboard

import time

events = []
start_time = time.time()


def on_move(x, y):
    t = time.time() - start_time
    events.append((t, 'MOVE', x, y))


def on_click(x, y, button, pressed):
    t = time.time() - start_time
    action = 'PRESSED' if pressed else 'RELEASED'
    events.append((t, f'MOUSE_{action}', x, y, str(button)))


def on_scroll(x, y, dx, dy):
    t = time.time() - start_time
    events.append((t, 'SCROLL', x, y, dx, dy))


def on_press(key):
    t = time.time() - start_time
    try:
        events.append((t, 'KEY_DOWN', key.char))
    except AttributeError:
        events.append((t, 'KEY_DOWN', str(key)))


def on_release(key):
    t = time.time() - start_time
    try:
        events.append((t, 'KEY_UP', key.char))
    except AttributeError:
        events.append((t, 'KEY_UP', str(key)))


if __name__ == "__main__":
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()

    mouse_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll
    )
    mouse_listener.start()

    # 4) Record only that region
    # record_screen("output.mp4", fps=60, region=region)

    # 5) Write logs

    print("Recording started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recording = False
        print("Stopping recording...")
        keyboard_listener.stop()
        mouse_listener.stop()
        print("Recording stopped. Check 'events_log.txt'.")
        with open('events_log.txt', 'w') as f:
            for ev in events:
                f.write(str(ev) + "\n")
        f.close()

    print("Done. Events saved to events_log.txt")
