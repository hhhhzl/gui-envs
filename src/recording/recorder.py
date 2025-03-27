from src.recording.screen import ScreenRecorder
from src.recording.mouse_key import MouseKeyRecorder
from src.utils import abspath
import time
from abc import ABC, abstractmethod


class Recorder(ABC):
    OPTION = ['both', 'screen', 'keyboard']

    def __init__(
            self,
            option='both',
            domain=None,
            task_description=None,
            path=abspath("data")
    ):
        assert(option in self.OPTION)
        self.domain = domain
        self.task_description = task_description
        self.path = path
        self.option = option

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def write_record(self, **kwargs):
        pass

    def record(self):
        if self.option == 'both':
            self.__record_both()
        elif self.option == 'screen':
            self.__record_screen()
        elif self.option == 'keyboard':
            self.__record_keyboard()

    def __record_both(self):
        start_time = time.time()
        # define keyboard and mouse
        mc = MouseKeyRecorder(start_time=start_time)
        sc = ScreenRecorder(filename=f'{"test"}.mp4', start_time=start_time, mkr=mc)
        mc.start()
        sc.start()

        print("\nRecording stopped.")
        mc.write_record(self.path, "test")
        print("Done. Events saved to")

    def __record_screen(self):
        ScreenRecorder(filename='test.mp4').start()

    def __record_keyboard(self):
        mc = MouseKeyRecorder().start()




