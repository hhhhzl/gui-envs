import pickle
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
from torch.autograd import Variable
from gui_envs.embeddings.language import LangEncoder
import numpy as np


class GUIAgent:
    def __init__(self, policy_path, language_instruction):
        self.policy = pickle.load(open(policy_path, 'rb'))
        self.policy.model.eval()
        self.language_emb = LangEncoder(device='cuda' if torch.cuda.is_available() else "cpu")
        self.instruction = language_instruction

    def act(self, obs):
        if type(obs) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
        obs = obs.unsqueeze(0)
        if self.instruction and self.language_emb:
            lang_emb = self.language_emb.forward(self.instruction)
            if type(lang_emb) is not torch.Tensor:
                lang_emb = Variable(torch.from_numpy(lang_emb).float(), requires_grad=False)
            obs = torch.cat([obs, lang_emb], -1)

        with torch.no_grad():
            act_pi = self.policy.model(obs)
        return act_pi
