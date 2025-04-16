import cv2
from typing import List, Optional
import subprocess
import time
from gui_envs.automations.automations import (
    scroll_mouse,
    click_mouse,
    press_key,
    release_key,
    move_mouse_position
)
import gym
import pyvirtualdisplay


class VirtualSimulatorGUI(object):
    def __init__(
            self,
            agent,
            env: gym.Env,
            app: str,
            screen_size:List,
            resolution:List,
            visible: bool = False,
            actions: Optional[List] = None,
            **kwargs
    ):
        self.app = app
        self.agent = agent
        self.env = env
        self.screen_size = tuple(screen_size)
        self.resolution = tuple(resolution)
        self.display = pyvirtualdisplay.Display(
            visible=visible,
            size=self.screen_size
        )
        if actions:
            self.mode = "playback"  # no need to interact
        else:
            self.mode = 'online'  # interact with real env
        self.actions = actions
        self.start_time = time.time()

    def run(self):
        self.display.start()
        time.sleep(2)
        try:
            self.__open_app()
        except Exception as e:
            print(f"Fail to open the app: {str(e)}.")
            return

        self.start_time = time.time()
        if self.mode == "playback":
            self.__run_play_back()
        elif self.mode == 'online':
            self.__run_online()
        else:
            raise "Not a correct mode"

    def __play_row_action(self, action, offset=0):
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

    def __run_play_back(self) -> None:
        """
        Replay from mouse_keyboard data
        :return:
        """
        for action in self.actions:
            self.__play_row_action(action)

    def __run_online(self, max_time=300) -> None:
        """
        RUN online with an env and agent policy
        :return:
        """
        state = self.env.reset()
        done = False
        while not done or time.time() - self.start_time < max_time:
            start = time.time()
            action = self.agent(state)
            state, reward, done, _ = self.env.step(action)
            inference_time = time.time() - start

        self.env.close()

    def __open_app(self):
        subprocess.run(["open", "-a", self.app.capitalize()])
        time.sleep(2)
