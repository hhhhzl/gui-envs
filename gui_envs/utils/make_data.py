# Read the /data, make demonstrations expert_data.pt
# expert_data:
#       state: image state, frame
#       next_state:
#       other_state: physical state (mouse position, cpu usage.... etc)
#       next_other_state:
#       action: action vector
#       ep_found_goal: whether achieved goal for this episode
# from typing import Dict
# import cv2
# import os
# import json
# import numpy as np
# import ast
# import bisect
# from tqdm import tqdm
# from gui_envs.envs.action_encode import one_hot_encode_event
# import pickle
#
#
# def find_closest_elements(arr1, arr2):
#     arr1_sorted = sorted(arr1)
#     result = []
#
#     for num in arr2:
#         pos = bisect.bisect_left(arr1_sorted, num)
#         if pos == 0:
#             closest = arr1_sorted[0]
#         elif pos == len(arr1_sorted):
#             closest = arr1_sorted[-1]
#         else:
#             left = arr1_sorted[pos - 1]
#             right = arr1_sorted[pos]
#             closest = left if (num - left) <= (right - num) else right
#         result.append(closest)
#     return result
#
#
# def get_video_frames_and_timestamps(video_path, initial_unix_timestamp):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("Could not open video file.")
#
#     frames = []
#     video_timestamps_ms = []
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
#         frames.append(frame)
#         video_timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
#
#     cap.release()
#     aligned_timestamps = [
#         initial_unix_timestamp + (ms / 1000.0)
#         for ms in video_timestamps_ms
#     ]
#     timestamps = {}
#     for i in range(len(aligned_timestamps)):
#         timestamps[aligned_timestamps[i]] = frames[i]
#     del frames
#     return aligned_timestamps, timestamps
#
#
# def get_keyboard_timestamps(keyboard_path):
#     with open(keyboard_path, 'r') as file:
#         lines = file.readlines()
#
#     keyboard_events = []
#     for line in lines:
#         stripped_line = line.strip()
#         if stripped_line:
#             try:
#                 parsed_list = ast.literal_eval(stripped_line)
#                 if isinstance(parsed_list, list):
#                     keyboard_events.append(parsed_list)
#             except (SyntaxError, ValueError):
#                 print(f"Skipping invalid line: {stripped_line}")
#
#     return keyboard_events, [row[0] for row in keyboard_events if row]
#
#
# def align_frames_actions(mapping_info: Dict):
#     states = []
#     actions = []
#
#     finished = 1 if mapping_info['finished'] else 0
#     video_info = mapping_info['video']
#     keyboard_info = mapping_info['keyboard']
#     task = mapping_info['task_description']
#     start_time = mapping_info['start_time']
#     if not video_info or not keyboard_info:
#         return
#
#     video_program_start_time = video_info['start_time']
#     video_record_start_time = video_program_start_time + video_info['offset']  # true recording start
#     video_timestamps, mappings = get_video_frames_and_timestamps(
#         root_path + mapping_info['path'] + "/video/" + mapping_info['video']['file_name'], video_record_start_time)
#     keyboard_events, keyboard_timestamps = get_keyboard_timestamps(
#         root_path + mapping_info['path'] + "/keyboard/" + mapping_info['keyboard']['file_name'])
#
#     video_times_dic = {}
#     keyboard_times_dic = {}
#     for time in video_timestamps:
#         video_times_dic[time] = []
#
#     for event in keyboard_events:
#         keyboard_times_dic[event[0]] = event
#
#     closest_frames = find_closest_elements(video_timestamps, keyboard_timestamps)
#
#     for i in range(len(closest_frames)):
#         video_times_dic[closest_frames[i]].append(keyboard_timestamps[i])
#
#     for frame_time in video_times_dic.keys():
#         state, action = None, None
#         if len(video_times_dic[frame_time]) == 0:
#             state = mappings[frame_time]
#             action = one_hot_encode_event(None, start_time)
#         else:
#             for each_keyboard_time in video_times_dic[frame_time]:
#                 state = mappings[frame_time]
#                 action = one_hot_encode_event(keyboard_times_dic[each_keyboard_time], start_time)
#
#         states.append(state)
#         actions.append(action)
#     del mappings
#     del video_times_dic
#     del keyboard_times_dic
#     assert len(states) == len(actions)
#     return {
#         'images': np.array(states),
#         'observations': task,
#         'actions': np.array(actions),
#         'finished': finished
#     }
#
#
# def make_dataset():
#     demo_path = []
#     for filename in tqdm(os.listdir(path), desc="Processing files"):
#         if filename.endswith(".json"):
#             filepath = os.path.join(path, filename)
#             with open(filepath, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#             file.close()
#             vectors_dic = align_frames_actions(data)
#             demo_path.append(vectors_dic)
#             del vectors_dic
#
#     with open('gui_demo_p.pickle', 'wb') as handle:
#         pickle.dump(demo_path, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# if __name__ == "__main__":
#     make_dataset()

from typing import Dict, List, Tuple, Optional
import cv2
import os
import json
import numpy as np
import ast
import bisect
from tqdm import tqdm
from gui_envs.envs.action_encode import one_hot_encode_event
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

root_path = "/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/"


def find_closest_elements(arr1: List[float], arr2: List[float]) -> List[float]:
    """Optimized version of finding closest elements between two arrays using bisect."""
    arr1_sorted = np.sort(arr1)  # Using numpy sort which is faster for large arrays
    arr1_sorted_list = arr1_sorted.tolist()  # Convert to list for bisect
    result = np.empty(len(arr2), dtype=np.float64)

    for i, num in enumerate(arr2):
        pos = bisect.bisect_left(arr1_sorted_list, num)
        if pos == 0:
            result[i] = arr1_sorted[0]
        elif pos == len(arr1_sorted):
            result[i] = arr1_sorted[-1]
        else:
            left = arr1_sorted[pos - 1]
            right = arr1_sorted[pos]
            result[i] = left if (num - left) <= (right - num) else right
    return result.tolist()


def get_video_frames_and_timestamps(video_path: str, initial_unix_timestamp: float) -> Tuple[
    List[float], Dict[float, np.ndarray]]:
    """Optimized video frame reading with pre-allocation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    video_timestamps_ms = []

    # Get approximate frame count for progress (not exact but helps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames.reserve(frame_count)
    # video_timestamps_ms.reserve(frame_count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        video_timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()

    # Vectorized timestamp calculation
    video_timestamps_ms_np = np.array(video_timestamps_ms, dtype=np.float64)
    aligned_timestamps = initial_unix_timestamp + (video_timestamps_ms_np / 1000.0)

    # Create dictionary more efficiently
    timestamps = dict(zip(aligned_timestamps, frames))

    return aligned_timestamps.tolist(), timestamps


def get_keyboard_timestamps(keyboard_path: str) -> Tuple[List[List], List[float]]:
    """Optimized keyboard timestamp reading."""
    try:
        with open(keyboard_path, 'r') as file:
            lines = file.readlines()

        keyboard_events = []
        keyboard_timestamps = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                parsed_list = ast.literal_eval(stripped_line)
                if isinstance(parsed_list, list) and parsed_list:
                    keyboard_events.append(parsed_list)
                    keyboard_timestamps.append(parsed_list[0])
            except (SyntaxError, ValueError):
                print(f"Skipping invalid line: {stripped_line}")

        return keyboard_events, keyboard_timestamps
    except Exception as e:
        print(f"Error reading keyboard file {keyboard_path}: {str(e)}")
        return [], []


def process_single_file(filename: str, path: str) -> Optional[Dict[str, np.ndarray]]:
    """Process a single file and return its data or None if failed."""
    filepath = os.path.join(path, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not data.get('video') or not data.get('keyboard'):
            return None

        finished = 1 if data['finished'] else 0
        task = data['task_description']
        start_time = data['start_time']

        video_info = data['video']
        keyboard_info = data['keyboard']

        # Path construction optimization
        video_path = os.path.join(root_path, data['path'], "video", video_info['file_name'])
        keyboard_path = os.path.join(root_path, data['path'], "keyboard", keyboard_info['file_name'])

        video_program_start_time = video_info['start_time']
        video_record_start_time = video_program_start_time + video_info['offset']

        video_timestamps, mappings = get_video_frames_and_timestamps(video_path, video_record_start_time)
        keyboard_events, keyboard_timestamps = get_keyboard_timestamps(keyboard_path)

        if not video_timestamps or not keyboard_timestamps:
            return None

        # Optimized dictionary creation
        video_times_dic = {time: [] for time in video_timestamps}
        keyboard_times_dic = {event[0]: event for event in keyboard_events}

        closest_frames = find_closest_elements(video_timestamps, keyboard_timestamps)

        # Optimized dictionary population
        for frame, k_time in zip(closest_frames, keyboard_timestamps):
            video_times_dic[frame].append(k_time)

        # Pre-allocate lists
        states = []
        actions = []
        # states.reserve(len(video_times_dic))
        # actions.reserve(len(video_times_dic))

        for frame_time, k_times in video_times_dic.items():
            state = mappings[frame_time]
            if not k_times:
                action = one_hot_encode_event(None, start_time)
            else:
                action = one_hot_encode_event(keyboard_times_dic[k_times[-1]], start_time)

            states.append(state)
            actions.append(action)

        # Convert to numpy arrays more efficiently
        return {
            'images': np.array(states),
            'observations': task,
            'actions': np.array(actions),
            'finished': finished
        }
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def make_dataset():
    """Optimized dataset creation with multiprocessing."""
    # hhhhzl/EduGetRicher/CMU/projects/GUI-VDILA/gui_envs/gui-envs
    path = '/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/GUI-VDILA/src/gui-envs/metadata/mapping'
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        process_func = partial(process_single_file, path=path)
        results = list(tqdm(
            pool.imap_unordered(process_func, json_files),
            total=len(json_files),
            desc="Processing files"
        ))

    demo_path = [result for result in results if result is not None]

    with open('gui_demo_p_v2.pickle', 'wb') as handle:
        pickle.dump(demo_path, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    make_dataset()
