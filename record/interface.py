import time
import tkinter as tk
from screen import record_screen
import multiprocessing


class Interface(object):

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.sidebar = tk.Toplevel()
        self.sidebar.overrideredirect(True)
        self.sidebar.attributes('-topmost', True)
        self.sidebar.geometry("450x50+500+700")
        self.sidebar.attributes('-alpha', 0.9)

        self.frame = tk.Frame(self.sidebar, bg="lightgray")
        self.frame.pack(fill='both', expand=True)

        self.btn_close = tk.Button(
            self.frame,
            text="Close",
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            bg="lightgray",
            activebackground="gray",
            command=self.on_capture_close_window
        )
        self.btn_close.pack(side="left", padx=10, pady=10)

        self.btn_full_screen = tk.Button(
            self.frame,
            text="Full Screen",
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            bg="lightgray",
            activebackground="gray",
            command=self.on_capture_window
        )
        self.btn_full_screen.pack(side="left", padx=10, pady=10)

        self.btn_capture_area = tk.Button(
            self.frame,
            text="Selected Area",
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            bg="lightgray",
            activebackground="gray",
            command=self.on_capture_area
        )
        self.btn_capture_area.pack(side="left", padx=10, pady=10)

        self.btn_record = tk.Button(
            self.frame,
            text="Record",
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            bg="white",
            activebackground="gray",
            command=self.on_capture_record
        )
        self.btn_record.pack(side="left", padx=20, pady=10)
        self._enable_sidebar_move()

    def start(self):
        self.root.mainloop()

    def _enable_sidebar_move(self):
        offset_x = tk.IntVar(value=0)
        offset_y = tk.IntVar(value=0)

        def click_mouse(event):
            offset_x.set(event.x)
            offset_y.set(event.y)

        def drag_mouse(event):
            x = self.sidebar.winfo_x() + event.x - offset_x.get()
            y = self.sidebar.winfo_y() + event.y - offset_y.get()
            self.sidebar.geometry(f"+{x}+{y}")

        self.frame.bind("<Button-1>", click_mouse)
        self.frame.bind("<B1-Motion>", drag_mouse)

    def on_capture_close_window(self):
        print("Window Closing...")
        self.root.quit()
        self.root.destroy()

    def on_capture_record(self):
        print("Recording...")
        self.sidebar.destroy()
        self.root.after(500, self._close_gui_and_start_recording)

    def _close_gui_and_start_recording(self):
        self.root.quit()
        self.root.destroy()
        multiprocessing.Process(target=record_screen, args=("test.mp4", 20)).start()

    def on_capture_window(self):
        """Handle full-screen capture logic here."""
        print("Capture Full Screen clicked!")

    def on_capture_area(self):
        """
        Create a full-screen overlay that lets the user
        click-and-drag to select a rectangular region.
        """
        print("Selected Area clicked!")
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        overlay.geometry(f"{screen_width}x{screen_height}+0+0")
        overlay.attributes('-alpha', 0.2)

        # Create a Canvas that fills the overlay
        canvas = tk.Canvas(overlay, bg="lightgray", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.configure(background='lightgray')
        canvas.attributes = {}

        rect_id = [None]  # store the Canvas rectangle object ID in a list so we can modify in nested functions
        start_xy = [0, 0]  # store the starting (x, y) of the mouse

        def on_mouse_down(event):
            start_xy[0] = event.x
            start_xy[1] = event.y
            rect_id[0] = canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline='red',
                width=2
            )

        def on_mouse_drag(event):
            if rect_id[0] is not None:
                canvas.coords(
                    rect_id[0],
                    start_xy[0],
                    start_xy[1],
                    event.x,
                    event.y
                )

        def on_mouse_up(event):
            if rect_id[0] is not None:
                x1, y1, x2, y2 = canvas.coords(rect_id[0])
                print(f"Selected rectangle = ({x1}, {y1}) to ({x2}, {y2})")

            overlay.destroy()

        canvas.bind("<Button-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_drag)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)


if __name__ == "__main__":
    Interface().start()
