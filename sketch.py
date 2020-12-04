import tkinter as tk
import sys

from gui.sketcher import Sketcher
from postprocessing.recognizer import Recognizer

if __name__ == "__main__":
    if len(sys.argv) == 3:
        recognizer = Recognizer(sys.argv[1])
        print(recognizer.recognize(sys.argv[2], "."))
    else:
        root = tk.Tk()
        root.geometry(f"{1600}x{800}+{10}+{10}")

        root.config(bg='white')
        root.title("Project")

        Sketcher(root, 10, 10, (1500, 600))
        root.mainloop()
