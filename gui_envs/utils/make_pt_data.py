# ===============================================================
# Read the /data, make demonstrations expert_data.pt
# expert_data:
#       obs: image state, frame
#       next_obs:
#       other_obs: physical state (mouse position, cpu usage.... etc) (optional)
#       next_obs: (optional)
#       actions: action vector
#       done: done for the episode
#       ep_found_goal: whether achieved goal for this episode
# ===============================================================
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
import torch
from gui_envs.envs.gui_env import _get_embedding, load_r3m_reproduce, ClipEnc
import torchvision.transforms as T
from PIL import Image
from gui_envs.embeddings.language import LangEncoder
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

root_path = "/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/"
path = '/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/GUI-VDILA/src/gui_envs/metadata/mapping'


def select_img_emb(img_emb, embedding_name='resnet50', device='cuda'):
    if img_emb == "clip":
        import clip
        model, cliptransforms = clip.load("RN50", device="cuda")
        embedding = ClipEnc(model)
        embedding.eval()
        embedding_dim = 1024
        transforms = cliptransforms
    elif (img_emb == "random") or (img_emb == ""):
        embedding, embedding_dim = _get_embedding(embedding_name=embedding_name, load_path=img_emb)
        transforms = T.Compose([T.Resize(256),
                                T.CenterCrop(224),
                                T.ToTensor(),  # ToTensor() divides by 255
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif img_emb == "r3m":
        rep = load_r3m_reproduce("r3m")
        rep.eval()
        embedding_dim = rep.module.outdim
        embedding = rep
        transforms = T.Compose([T.Resize(256),
                                T.CenterCrop(224),
                                T.ToTensor()])  # ToTensor() divides by 255
    else:
        raise NameError("Invalid Model")
    embedding.eval()

    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = device
    embedding.to(device=device)

    embedding, embedding_dim = embedding, embedding_dim
    return embedding, embedding_dim, transforms, device


class Embedding:
    def __init__(self, img_emb):
        # define embedding
        self.img_embedding, self.embedding_dim, self.transforms, self.device = select_img_emb(img_emb, embedding_name='resnet50',device='cuda')
        self.language_emb = LangEncoder(device='cuda' if torch.cuda.is_available() else "cpu")
        self.img_emb = img_emb

    def forward_img(self, obs):
        inp = self.transforms(Image.fromarray(obs.astype(np.uint8))).reshape(-1, 3, 224, 224)
        if "r3m" in self.img_emb:
            # R3M Expects input to be 0-255, preprocess makes 0-1
            inp *= 255.0
        inp = inp.to(self.device)
        with torch.no_grad():
            emb = self.img_embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def forward_lang(self, instruction):
        lang_emb = self.language_emb.forward(instruction)
        return lang_emb


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
    cap = cv2.VideoCapture(video_path.replace("gui-envs", "gui_envs"))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    video_timestamps_ms = []

    # Get approximate frame count for progress (not exact but helps)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        with open(keyboard_path.replace("gui-envs", "gui_envs"), 'r') as file:
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


def process_single_file(filename: str, path: str, emb_cls) -> Optional[Dict[str, np.ndarray]]:
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

        if len(video_timestamps) == 0 or len(keyboard_timestamps) == 0:
            return None
        # Optimized dictionary creation
        video_times_dic = {time: [] for time in video_timestamps}
        keyboard_times_dic = {event[0]: event for event in keyboard_events}

        closest_frames = find_closest_elements(video_timestamps, keyboard_timestamps)

        # Optimized dictionary population
        for frame, k_time in zip(closest_frames, keyboard_timestamps):
            video_times_dic[frame].append(k_time)

        # Pre-allocate lists
        obs, next_obs, actions, done, ep_found_goal = [], [], [], [], []
        other_obs, next_other_obs = [], [] # optional
        lang_emb = None
        for i, (frame_time, k_times) in enumerate(video_times_dic.items()):
            # ----- obs -------#
            ob = mappings[frame_time]
            ob = emb_cls.forward_img(ob)
            if type(ob) is not torch.Tensor:
                ob = Variable(torch.from_numpy(ob).float(), requires_grad=False)

            ob = ob.unsqueeze(0)
            if task and emb_cls.language_emb:
                if lang_emb is None:
                    lang_emb = emb_cls.forward_lang(task)
                    if type(lang_emb) is not torch.Tensor:
                        lang_emb = Variable(torch.from_numpy(lang_emb).float(), requires_grad=False)
                ob = torch.cat([ob, lang_emb], -1)

            # ----- actions -------#
            if len(k_times) == 0:
                action = one_hot_encode_event(None, start_time)
            else:
                action = one_hot_encode_event(keyboard_times_dic[k_times[-1]], start_time)
            action = torch.as_tensor(action, dtype=torch.float32)

            obs.append(ob)
            actions.append(action)
            ep_found_goal.append(finished)
            if i != 0:
                next_obs.append(ob)
            if i == len(video_times_dic) - 1:
                next_obs.append(ob)
                done.append(1)
            else:
                done.append(0)
        assert len(obs) == len(actions) == len(next_obs) == len(done) == len(ep_found_goal)
        return_dict = {
            "obs": torch.stack(obs),  # (T, D_obs)
            "next_obs": torch.stack(next_obs),  # (T, D_obs)
            "actions": torch.stack(actions),  # (T, A)
            "done": torch.tensor(done),  # (T,)
            "ep_found_goal": torch.tensor(ep_found_goal),  # (T,)
        }
        if len(other_obs) != 0 and len(next_other_obs) != 0 and len(other_obs) == len(other_obs):
            assert len(other_obs) == len(obs)
            return_dict["other_obs"] = torch.stack(other_obs)
            return_dict["next_other_obs"] = torch.stack(next_other_obs)
        return return_dict
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def make_pt_dataset(
    img_emb: str = "r3m",
    language_emb: str = "",
):
    """Optimized dataset creation with multiprocessing."""
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    emb_cls = Embedding(img_emb)
    worker = partial(process_single_file, path=path, emb_cls=emb_cls)

    with Pool(max(1, cpu_count() - 10)) as pool:
        results: List[Optional[Dict[str, torch.Tensor]]] = list(
            tqdm(pool.imap_unordered(worker, json_files),
                 total=len(json_files), desc="Processing")
        )

    valid_res = [r for r in results if r is not None]
    if not valid_res:
        raise RuntimeError("Parsing Error")

    merged: Dict[str, List[torch.Tensor]] = {k: [] for k in valid_res[0].keys()}
    for r in valid_res:
        for k, v in r.items():
            merged[k].append(v)

    save_name = f"gui_{img_emb}_{len(json_files)}.pt"
    dataset = {k: torch.cat(v, dim=0) for k, v in merged.items()}
    torch.save(dataset, save_name)
    print(f"Saved to {save_name}. ")
    for k, v in dataset.items():
        print(f"  {k:<12s}: {tuple(v.shape)}")


if __name__ == "__main__":
    make_pt_dataset()
