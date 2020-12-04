import tkinter as tk
from enum import Enum, auto, unique
from tkinter import messagebox

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw

from postprocessing.recognizer import Recognizer


@unique
class MouseState(Enum):
    PRESSED = auto()
    RELEASED = auto()


class Sketcher:
    DEFAULT_MODEL = "model/models/convolutional.h5"
    LINE_WIDTH = 8

    def __init__(self, parent, x_position, y_position, size):
        self.parent = parent
        self.x_position = x_position
        self.y_position = y_position

        self.recognizer = Recognizer(Sketcher.DEFAULT_MODEL)

        self.size = size

        self.mouse_state = MouseState.RELEASED
        self.previous_coordinates = None

        self.drawing_area = tk.Canvas(self.parent, width=size[0], height=size[1])
        self.drawing_area.pack(padx=30, pady=40)
        self.add_bindings()

        self.button_recognize = tk.Button(self.parent, text="Textify!", width=10, bg='white', command=self.recognize)

        self.button_recognize.pack(padx=20, pady=10)

        self.button_clear = tk.Button(self.parent, text="Clear!", width=10, bg='white', command=self.clear)
        self.button_clear.pack(padx=20, pady=10)

        self.draw = None
        self.image = None

        self.points = []

        self.clear()

    def add_bindings(self):
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.mouse_press)
        self.drawing_area.bind("<ButtonRelease-1>", self.mouse_release)


    def recognize(self):
        if self.points:
            self.draw.line(self.points, ImageColor.getrgb("black"),
                           width=Sketcher.LINE_WIDTH, joint="curve")
            self.points.clear()

        filename = "images/temp.png"
        self.image.save(filename)

        result = self.recognizer.recognize("temp.png", "images")
        print(result)
        messagebox.showinfo("Result", result)

    def clear(self):
        self.drawing_area.delete("all")
        self.points.clear()
        self.image = Image.new("RGB", self.size, ImageColor.getrgb("white"))
        self.draw = ImageDraw.Draw(self.image)

    def mouse_release(self, event):
        if self.points:
            self.draw.line([*self.points, self.previous_coordinates], ImageColor.getrgb("black"),
                           width=Sketcher.LINE_WIDTH, joint="curve")
            self.points.clear()

        self.mouse_state = MouseState.RELEASED

    def mouse_press(self, event):
        self.mouse_state = MouseState.PRESSED
        self.previous_coordinates = None

    def motion(self, event):
        x, y = event.x, event.y
        if self.mouse_state == MouseState.PRESSED:
            if self.previous_coordinates:
                event.widget.create_line(*self.previous_coordinates, x, y, smooth="true",
                                         width=Sketcher.LINE_WIDTH,
                                         fill="black", capstyle=tk.ROUND)
                self.points.append(self.previous_coordinates)
        self.previous_coordinates = x, y
