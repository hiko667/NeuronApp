import tkinter as tk
from logic.machine import Machine
from tkinter import filedialog
import os

class Interface():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neuron App!")
        menubar = tk.Menu(self.root)
        file = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label ='File', menu = file)
        file.add_command(label ='Open Project', command = None)
        file.add_command(label ='Save Project', command = None)
        file.add_separator()
        file.add_command(label ='Exit', command = self.root.destroy)

        edit = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label ='Dataset', menu = edit)
        edit.add_command(label ='Edit', command = None)
        edit.add_command(label ='Load File', command = self.on_load_file)

        machine = tk.Menu(menubar, tearoff = 0)
        menubar.add_cascade(label ='Machine', menu = machine)
        machine.add_command(label ='Perceptron', command = None)
        machine.add_command(label ='Adeline', command = None)
        self.root.config(menu = menubar)

        self.machine = Machine()
        self.root.mainloop()
    def on_load_file(self):
        filename = filedialog.askopenfilename(initialdir = os.path.abspath(os.sep), title="Select file", filetypes=(
            ("Tab separated files","*.tab*"),
            ("all files", "*.*")))