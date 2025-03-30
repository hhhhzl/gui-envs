import numpy as np

EVENT_TYPES = ["MOVE", "MOUSE_PRESSED", "MOUSE_RELEASED", "SCROLL", "KEY_DOWN", "KEY_UP"]


def one_hot_encode_event(event):
    event_type = event[1]

    action_vector = np.zeros(len(EVENT_TYPES))
    action_vector[EVENT_TYPES.index(event_type)] = 1

    x, y, dx, dy, key_button = 0, 0, 0, 0, 0

    if event_type == "MOVE":
        x, y = event[2], event[3]
    elif "MOUSE" in event_type:
        x, y = event[2], event[3]
        key_button = 1 if "left" in event[-1] else 2  # Example encoding for buttons
    elif event_type == "SCROLL":
        x, y, dx, dy = event[2], event[3], event[4], event[5]
    elif "KEY" in event_type:
        key_button = hash(event[2]) % 1000  # Encode key uniquely

    return np.concatenate((action_vector, [event[-1], x, y, dx, dy, key_button]))
