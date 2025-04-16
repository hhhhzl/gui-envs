import time
import pyautogui
from pynput.keyboard import Key


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
    except Exception as e:
        print(f"Press mouse to ({x}, {y}) Failed: {str(e)}")


def click_mouse(x, y, button, pressed, delta_t):
    simulate_velocity(delta_t)
    if pressed:
        pyautogui.mouseDown(x, y, button=button.lower())
    else:
        pyautogui.mouseUp(x, y, button=button.lower())


def scroll_mouse(x, y, dx, dy, delta_t):
    simulate_velocity(delta_t)
    pyautogui.scroll(dy, x, y)
