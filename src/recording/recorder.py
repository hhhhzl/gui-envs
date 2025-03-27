from src.recording.screen import ScreenRecorder
from src.recording.mouse_key import MouseKeyRecorder
from src.utils import abspath
import time
from abc import ABC, abstractmethod
import os


class RecorderEngine(ABC):
    OPTION = ['both', 'screen', 'keyboard']

    def __init__(
            self,
            option='both',
            domain=None,
            task_description=None,
            path=abspath("data"),
    ):
        assert(option in self.OPTION)
        self.domain = domain
        self.task_description = task_description
        self.path = path
        self.option = option
        self.mc = None
        self.sc = None

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
        pass

    def __record_both(self):
        start_time = time.time()
        file_name = f"{int(start_time * 1000)}"

        # define keyboard and mouse
        self.mc = MouseKeyRecorder(start_time=start_time)
        self.sc = ScreenRecorder(filename=f'{self.path}/video/{file_name}.mp4', start_time=start_time, mkr=self.mc)
        self.mc.start()
        self.sc.start()

        print("\nRecording stopped.")
        self.mc.write_record(path=f'{self.path}/keyboard', file_name=file_name)
        print("Done. Events saved to")

    def __record_screen(self):
        start_time = time.time()
        file_name = f"{int(start_time * 1000)}"
        self.sc = ScreenRecorder(filename=f'{self.path}/video/{file_name}.mp4').start()

    def __record_keyboard(self):
        start_time = time.time()
        file_name = f"{int(start_time * 1000)}"
        self.mc = MouseKeyRecorder().start()
        self.mc.write_record(path=f'{self.path}/keyboard', file_name=file_name)




