import gym
import torch
from src.simulator.monitor import RealSimulatorGUI
from src.envs.gui_env import StateEmbedding, GUIPixelObs
from src.envs.gym_env import GymEnv
from src.simulator.agent import GUIAgent


def env_constructor(
        env_name,
        device='cuda',
        image_width=256,
        image_height=256,
        camera_name=None,
        embedding_name='resnet50',
        pixel_based=True,
        render_gpu_id=0,
        load_path="",
        proprio=False,
        lang_cond=False,
        gc=False
):
    # If pixel based will wrap in a pixel observation wrapper
    if pixel_based:
        e = gym.make(env_name)
        # Wrap in pixel observation wrapper
        e = GUIPixelObs(
            e,
            width=image_width,
            height=image_height,
            camera_name=camera_name,
            device_id=render_gpu_id
        )
        # Wrapper which encodes state in pretrained model
        e = StateEmbedding(
            e,
            embedding_name=embedding_name,
            device=device,
            load_path=load_path,
            proprio=proprio,
            camera_name=camera_name,
            env_name=env_name
        )
        e = GymEnv(e)
    else:
        print("Only supports pixel based")
        assert (False)
    return e


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = "r3m"
    env_input = {
        "env_name": "CartPole-v1",
        "device": device,
        "image_width": 256,
        "image_height": 256,
        "camera_name": "default",
        "embedding_name": "resnet50",
        "pixel_based": True,
        "render_gpu_id": 0,
        "load_path": model,
        "proprio": 0,
        "lang_cond": False,
        "gc": False
    }
    env = env_constructor(**env_input)
    agent = GUIAgent("policy_338.pickle")
    simulator = RealSimulatorGUI(
        agent=agent,
        env=env,
        app='Slack'
    )
    simulator.run()









