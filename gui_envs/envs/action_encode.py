import numpy as np

EVENT_TYPES = [
    "NULL",  # New null action type
    "MOVE",
    "MOUSE_PRESSED",
    "MOUSE_RELEASED",
    "SCROLL",
    "KEY_DOWN",
    "KEY_UP"
]


def one_hot_encode_event(event, start_time):
    if event is None:
        action_vector = np.zeros(len(EVENT_TYPES))
        action_vector[EVENT_TYPES.index("NULL")] = 1
        return np.concatenate((action_vector, [0, 0, 0, 0, 0, 0]))

    event_type = event[1]
    delta_t = event[0] - start_time
    action_vector = np.zeros(len(EVENT_TYPES))
    action_vector[EVENT_TYPES.index(event_type)] = 1

    x, y, dx, dy, key_button = 0, 0, 0, 0, 0

    if event_type == "MOVE":
        x, y = event[2], event[3]
    elif "MOUSE" in event_type:
        x, y = event[2], event[3]
        key_button = 1 if "left" in event[-1] else 2
    elif event_type == "SCROLL":
        x, y, dx, dy = event[2], event[3], event[4], event[5]
    elif "KEY" in event_type:
        key_button = hash(event[2]) % 1000

    return np.concatenate((action_vector, [delta_t, x, y, dx, dy, key_button]))


def convert_back(action):
    action = np.array(action)[0]
    event_probs = action[:len(EVENT_TYPES)]
    print(event_probs)
    params = action[len(EVENT_TYPES):]

    # Get the most likely event type
    event_idx = np.argmax(event_probs)
    event_type = EVENT_TYPES[event_idx]

    # Extract parameters
    delta_t, x, y, dx, dy, key_button = params

    result = {
        "event_type": event_type,
        "delta_t": delta_t,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "key": key_button
    }
    return result
