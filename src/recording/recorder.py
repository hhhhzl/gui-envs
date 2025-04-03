from src.recording.screen import ScreenRecorder
from src.recording.mouse_key import MouseKeyRecorder
from src.utils import abspath
import time
from abc import ABC
import os
from typing import Optional
import json


class RecorderEngine(ABC):
    OPTION = ['both', 'screen', 'keyboard']

    def __init__(
            self,
            option='both',
            domain=None,
            task_description=None,
            path=abspath("metadata"),
            selected_area=None,
    ):
        assert(option in self.OPTION)
        self.domain = domain
        self.task_description = task_description
        self.path = path
        self.option = option
        self.mc: Optional[MouseKeyRecorder] = None
        self.sc: Optional[ScreenRecorder] = None
        self.selected_area = selected_area
        self.file_name = None
        self.start_time = None

        for folder in ['video', 'mapping', 'keyboard']:
            exists = os.path.exists(f"{self.path}/{folder}")
            if not exists:
                os.makedirs(f"{self.path}/{folder}")

    def start(self):
        if self.option == 'both':
            self.__record_both()
        elif self.option == 'screen':
            self.__record_screen()
        elif self.option == 'keyboard':
            self.__record_keyboard()

    def write_record(self, **kwargs):
        mapping = {
            "domain": self.domain,
            "task_description": self.task_description,
            "path": self.path,
            "start_time": self.start_time,
            "finished": None, # human labeled data
        }
        if self.sc:
            mapping['video'] = {
                'file_name': f"{self.file_name}.mp4",
                'screen_size': self.sc.screen_size,
                'resolution': [self.sc.resolution[0], self.sc.resolution[1]],
                'selected_area': self.selected_area,
                'fps': self.sc.fps,
                'duration': self.sc.duration,
                'start_time': self.sc.start_time,
                'offset':self.sc.time_offset,
                'frames': self.sc.frames,
            }
        if self.mc:
            mapping['keyboard'] = {
                'file_name': f"{self.file_name}.txt",
                'start_time': self.mc.start_time,
                'action_numbers': len(self.mc.events)
            }
        mapping['audio'] = None # not support audio now
        try:
            with open(f'{self.path}/mapping/{self.file_name}.json', "w") as file:
                json.dump(mapping, file, indent=4)
            print("Mapping File Saved")
        except Exception as e:
            print(f"Mapping File Save Failed: {str(e)}")

    def __record_both(self):
        self.start_time = time.time()
        self.file_name = f"{int(self.start_time * 1000)}"

        # define keyboard and mouse
        self.mc = MouseKeyRecorder(start_time=self.start_time)
        self.sc = ScreenRecorder(filename=f'{self.path}/video/{self.file_name}.mp4', start_time=self.start_time, mkr=self.mc)
        self.mc.start()
        self.sc.start()

        print("\nRecording stopped.")
        self.sc.write_record()
        self.write_record()
        self.mc.write_record(path=f'{self.path}/keyboard', file_name=self.file_name)

    def __record_screen(self):
        self.start_time = time.time()
        self.file_name = f"{int(self.start_time * 1000)}"
        self.sc = ScreenRecorder(filename=f'{self.path}/video/{self.file_name}.mp4').start()

    def __record_keyboard(self):
        self.start_time = time.time()
        self.file_name = f"{int(self.start_time * 1000)}"
        self.mc = MouseKeyRecorder(start_time=self.start_time).start()
        self.mc.write_record(path=f'{self.path}/keyboard', file_name=self.file_name)




