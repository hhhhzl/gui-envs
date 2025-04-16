import time
import pyautogui
from pynput.keyboard import Key

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1


def simulate_velocity(delta_t):
    if not delta_t:
        time.sleep(0.2)
    else:
        time.sleep(delta_t)


def press_key(key, delta_t):
    simulate_velocity(delta_t)
    try:
        pyautogui.keyDown(key)
    except Exception as e:
        print(f"Press key {str(key)} Failed: {str(e)}")


def release_key(key, delta_t):
    simulate_velocity(delta_t)
    try:
        pyautogui.keyUp(key)
    except Exception as e:
        print(f"Release key {str(key)} Failed: {str(e)}")


def move_mouse_position(x, y, delta_t):
    simulate_velocity(delta_t)
    try:
        pyautogui.moveTo(x, y)
        print(f"Move mouse to ({x}, {y})")
    except Exception as e:
        print(f"Move mouse to ({x}, {y}) Failed: {str(e)}")


def click_mouse(x, y, button, pressed, delta_t):
    simulate_velocity(delta_t)
    if pressed:
        pyautogui.mouseDown(x, y, button=button.lower())
    else:
        pyautogui.mouseUp(x, y, button=button.lower())


def scroll_mouse(x, y, dx, dy, delta_t):
    simulate_velocity(delta_t)
    pyautogui.scroll(dy, x, y)


if __name__ == "__main__":
    import random

    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")

    while True:
        x = random.randint(100, screen_width - 100)
        y = random.randint(100, screen_height - 100)
        delay = random.uniform(0.5, 2.0)
        print(f"Moving to ({x}, {y}) with delay {delay}")
        move_mouse_position(x, y, delay)
