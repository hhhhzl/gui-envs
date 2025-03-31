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


def get_video_timestamps(video_path, fps):
    cap = cv2.VideoCapture(video_path)

    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    while (cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            break

    cap.release()
    return timestamps


def align_frames_actions(mapping_info: Dict):
    state = []
    action = []
    finished = mapping_info['finished']
    video_info = mapping_info['video']
    keyboard_info = mapping_info['keyboard']
    if not video_info or not keyboard_info:
        return

    video_program_start_time = video_info['start_time']
    video_record_start_time = video_program_start_time + video_info['offset']  # true recording start
    video_record_end_time = video_record_start_time + video_info['duration']
