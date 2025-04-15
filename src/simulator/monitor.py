import cv2
from typing import List, Optional
import subprocess
import time
import gym


class RealSimulatorGUI(object):
    def __init__(
            self,
            agent,
            env: gym.Env,
            app: str,
            monitor_id=0,
            actions: Optional[List] = None,
            **kwargs
    ):
        self.app = app
        self.agent = agent
        self.env = env
        self.monitor_id = monitor_id
        if actions:
            self.mode = "playback"  # no need to interact
        else:
            self.mode = 'online'  # interact with real env
        self.actions = actions
        self.start_time = time.time()

    def run(self):
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


    def __run_play_back(self) -> None:
        """
        Replay from mouse_keyboard data
        :return:
        """
        for action in self.actions:
            self.env.step(action)

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
