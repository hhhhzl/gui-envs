# Read the /data, make demonstrations expert_data.pt
# expert_data:
#       state: image state, frame
#       next_state:
#       other_state: physical state (mouse position, cpu usage.... etc)
#       next_other_state:
#       action: action vector
#       ep_found_goal: whether achieved goal for this episode
from typing import Dict
import cv2
import os
import json
import numpy as np
import ast
import bisect
from tqdm import tqdm
from src.envs.action_encode import one_hot_encode_event
import pickle


def find_closest_elements(arr1, arr2):
    arr1_sorted = sorted(arr1)
    result = []

    for num in arr2:
        pos = bisect.bisect_left(arr1_sorted, num)
        if pos == 0:
            closest = arr1_sorted[0]
        elif pos == len(arr1_sorted):
            closest = arr1_sorted[-1]
        else:
            left = arr1_sorted[pos - 1]
            right = arr1_sorted[pos]
            closest = left if (num - left) <= (right - num) else right
        result.append(closest)
    return result


def get_video_frames_and_timestamps(video_path, initial_unix_timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frames = []
    video_timestamps_ms = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        video_timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()
    aligned_timestamps = [
        initial_unix_timestamp + (ms / 1000.0)
        for ms in video_timestamps_ms
    ]
    timestamps = {}
    for i in range(len(aligned_timestamps)):
        timestamps[aligned_timestamps[i]] = frames[i]
    del frames
    return aligned_timestamps, timestamps


def get_keyboard_timestamps(keyboard_path):
    with open(keyboard_path, 'r') as file:
        lines = file.readlines()

    keyboard_events = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            try:
                parsed_list = ast.literal_eval(stripped_line)
                if isinstance(parsed_list, list):
                    keyboard_events.append(parsed_list)
            except (SyntaxError, ValueError):
                print(f"Skipping invalid line: {stripped_line}")

    return keyboard_events, [row[0] for row in keyboard_events if row]


def align_frames_actions(mapping_info: Dict):
    states = []
    actions = []

    finished = 1 if mapping_info['finished'] else 0
    video_info = mapping_info['video']
    keyboard_info = mapping_info['keyboard']
    task = mapping_info['task_description']
    start_time = mapping_info['start_time']
    if not video_info or not keyboard_info:
        return

    video_program_start_time = video_info['start_time']
    video_record_start_time = video_program_start_time + video_info['offset']  # true recording start
    video_timestamps, mappings = get_video_frames_and_timestamps(
        root_path + mapping_info['path'] + "/video/" + mapping_info['video']['file_name'], video_record_start_time)
    keyboard_events, keyboard_timestamps = get_keyboard_timestamps(
        root_path + mapping_info['path'] + "/keyboard/" + mapping_info['keyboard']['file_name'])

    video_times_dic = {}
    keyboard_times_dic = {}
    for time in video_timestamps:
        video_times_dic[time] = []

    for event in keyboard_events:
        keyboard_times_dic[event[0]] = event

    closest_frames = find_closest_elements(video_timestamps, keyboard_timestamps)

    for i in range(len(closest_frames)):
        video_times_dic[closest_frames[i]].append(keyboard_timestamps[i])

    for frame_time in video_times_dic.keys():
        state, action = None, None
        if len(video_times_dic[frame_time]) == 0:
            state = mappings[frame_time]
            action = one_hot_encode_event(None, start_time)
        else:
            for each_keyboard_time in video_times_dic[frame_time]:
                state = mappings[frame_time]
                action = one_hot_encode_event(keyboard_times_dic[each_keyboard_time], start_time)

        states.append(state)
        actions.append(action)
    del mappings
    del video_times_dic
    del keyboard_times_dic
    assert len(states) == len(actions)
    return {
        'images': np.array(states),
        'observations': task,
        'actions': np.array(actions),
        'finished': finished
    }


def make_dataset():
    demo_path = []
    for filename in tqdm(os.listdir(path), desc="Processing files"):
        if filename.endswith(".json"):
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            file.close()
            vectors_dic = align_frames_actions(data)
            demo_path.append(vectors_dic)
            del vectors_dic

    with open('gui_demo.pickle', 'wb') as handle:
        pickle.dump(demo_path, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    make_dataset()
