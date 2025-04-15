import pickle
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
from torch.autograd import Variable
from src.embeddings.language import LangEncoder


class GUIAgent:
    def __init__(self, policy_path):
        self.policy = pickle.load(open(policy_path, 'rb'))
        self.policy.eval()
        self.language_emb = LangEncoder(device='cuda' if torch.cuda.is_available() else "cpu")

    def act(self, obs, language=None):
        if language and self.language_emb:
            language = self.language_emb.forward(language)
            if type(language) is not torch.Tensor:
                language = Variable(torch.from_numpy(language).float(), requires_grad=False).cuda()
            obs = torch.cat([obs, language], -1)

        with torch.no_grad():
            action = self.policy(obs)
        print(action)
        return action
