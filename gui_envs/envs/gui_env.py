"""
Build on https://github.com/facebookresearch/r3m/blob/eval/evaluation/r3meval/utils/obs_wrappers.py
"""
import numpy as np
import gym
from gym.spaces.box import Box
import omegaconf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from gym import spaces
import os
from os.path import expanduser
import hydra
import gdown
import copy
from tqdm import tqdm
import cv2
from mss import mss
from gui_envs.automations.automations import (
    scroll_mouse,
    click_mouse,
    press_key,
    release_key,
    move_mouse_position
)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight",
              "l2dist", "bs"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    # config.agent["_target_"] = "gui_envs.embeddings.pvrs.r3m.R3M"
    config.agent["_target_"] = "src.pvrs.models.r3m.r3m.R3M"
    config["device"] = device

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 0
    return config.agent


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict


def load_r3m_reproduce(modelid):
    home = os.path.join(expanduser("~"), ".r3m")
    if modelid == "r3m":
        foldername = "original_r3m"
        modelurl = 'https://drive.google.com/uc?id=1jLb1yldIMfAcGVwYojSQmMpmRM7vqjp9'
        configurl = 'https://drive.google.com/uc?id=1cu-Pb33qcfAieRIUptNlG1AQIMZlAI-q'
    elif modelid == "r3m_noaug":
        foldername = "original_r3m_noaug"
        modelurl = 'https://drive.google.com/uc?id=1k_ZlVtvlktoYLtBcfD0aVFnrZcyCNS9D'
        configurl = 'https://drive.google.com/uc?id=1hPmJwDiWPkd6GGez6ywSC7UOTIX7NgeS'
    elif modelid == "r3m_nol1":
        foldername = "original_r3m_nol1"
        modelurl = 'https://drive.google.com/uc?id=1LpW3aBMdjoXsjYlkaDnvwx7q22myM_nB'
        configurl = 'https://drive.google.com/uc?id=1rZUBrYJZvlF1ReFwRidZsH7-xe7csvab'
    elif modelid == "r3m_nolang":
        foldername = "original_r3m_nolang"
        modelurl = 'https://drive.google.com/uc?id=1FXcniRei2JDaGMJJ_KlVxHaLy0Fs_caV'
        configurl = 'https://drive.google.com/uc?id=192G4UkcNJO4EKN46ECujMcH0AQVhnyQe'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device(device))['r3m'])

    rep.load_state_dict(r3m_state_dict)
    return rep


def _get_embedding(embedding_name='resnet34', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim


class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """

    def __init__(self, env, embedding_name=None, device='cuda', load_path="", proprio=0, camera_name=None,
                 env_name=None, language=True):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        if load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == ""):
            embedding, embedding_dim = _get_embedding(embedding_name=embedding_name, load_path=load_path)
            self.transforms = T.Compose([T.Resize(256),
                                         T.CenterCrop(224),
                                         T.ToTensor(),  # ToTensor() divides by 255
                                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif load_path == "r3m":
            rep = load_r3m_reproduce("r3m")
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                                         T.CenterCrop(224),
                                         T.ToTensor()])  # ToTensor() divides by 255
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.embedding_dim + self.proprio + 768 if language else 0,))

    def observation(self, observation):
        # INPUT SHOULD BE [0,255]
        if self.embedding is not None:
            inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                # R3M Expects input to be 0-255, preprocess makes 0-1
                inp *= 255.0
            inp = inp.to(self.device)
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            # IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs, finetune=False, return_numpy=True):
        # INPUT SHOULD BE [0,255]
        inp = []
        for o in obs:
            i = self.transforms(Image.fromarray(o.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                # R3M Expects input to be 0-255, preprocess makes 0-1
                i *= 255.0
            inp.append(i)

        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
            with torch.no_grad():
                emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()

    def start_finetuning(self):
        self.start_finetune = True


class GUIPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, monitor=0, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(13, ))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        self.sct = mss()
        self.monitor = self.sct.monitors[monitor]

    def get_image(self):
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return img[:, :, :3]

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()

    def step(self, action, offset=0):
        event_time, delta_t, event_type, *args = action
        delta_t = delta_t - offset  # offset to the inference time
        if delta_t < 0:
            delta_t = 0

        if event_type == "MOVE":
            move_mouse_position(*args, delta_t)
        elif event_type.startswith("MOUSE_"):
            if 'press' in event_type.lower():
                click_mouse(*args, True, delta_t)
            else:
                click_mouse(*args, False, delta_t)
        elif event_type == "SCROLL":
            scroll_mouse(*args, delta_t)
        elif event_type == "KEY_DOWN":
            press_key(*args, delta_t)
        elif event_type == "KEY_UP":
            release_key(*args, delta_t)

        return self.get_image(), 0, False

    def reset(self):
        return self.get_image()
