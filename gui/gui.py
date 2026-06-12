import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import time

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.machine import Machine, AdalineMachine


class Interface():
    def __init__(self):
        self.initialize_machine()
        self.path = None
        self.root = tk.Tk()
        self.root.title("Neuron App!")

        #=====Menu=====
        menubar = tk.Menu(self.root)

        file = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file)
        file.add_command(label='Open Project', command=None)
        file.add_command(label='Save Project', command=None)
        file.add_separator()
        file.add_command(label='Exit', command=self.root.destroy)

        edit = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Dataset', menu=edit)
        edit.add_command(label='Load File', command=self.on_load_file)
        edit.add_command(label='Edit',      command=None)

        machine_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Machine', menu=machine_menu)
        machine_menu.add_command(label='Perceptron', command=self.on_perceptron)
        machine_menu.add_command(label='Adaline',    command=self.on_adeline)

        self.root.config(menu=menubar)

        #Control Panel
        ctrl = tk.Frame(self.root, width=180, padx=8, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl, text="Steps per click").pack(anchor='w')
        self.steps_var = tk.IntVar(value=1)

        tk.Spinbox(ctrl, from_=1, to=10_000, textvariable=self.steps_var, width=10).pack(anchor='w')
        tk.Button(ctrl, text="▶ Learn (n steps)", command=self.on_learn_steps, width=18).pack(pady=(10, 2))
        tk.Button(ctrl, text="▷ Learn slowly (n steps)", command=self.on_learn_slowly, width=18).pack(pady=(10))
        tk.Button(ctrl, text="⫸ Learn for treshold", command=self.learn_until, width=18).pack(pady=(10))

        tk.Button(ctrl, text="⟳ Reset weights", command=self.on_reset, width=18).pack(pady=2)
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
        
        tk.Label(ctrl, text="Target error").pack(anchor='w')
        self.error_expected = tk.DoubleVar(value=0.1)
        tk.Entry(ctrl, textvariable=self.error_expected, width=10).pack(anchor='w')
        tk.Button(ctrl, text="Update target error", command=self.on_update_error, width=18).pack(pady=2)


        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)

        self.info_text = tk.StringVar(value="No data loaded.")
        tk.Label(ctrl, textvariable=self.info_text, justify='left', wraplength=165, fg='#333').pack(anchor='w')

        #matplotlib
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6.5, 8.0), dpi=96)
        self.ax     = self.fig.add_subplot(211)  
        self.ax_err = self.fig.add_subplot(212)  
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.draw_plot()
        self.root.mainloop()

    #machine operatons
    def initialize_machine(self):
        self.machine = Machine()

    def on_reset(self):
        self.machine.reset()
        self.update_info()
        self.draw_plot()

    #menu callbacks
    def on_load_file(self):
        filename = filedialog.askopenfilename(
            # initialdir=os.path.abspath(os.sep),
            title="Select file",
            filetypes=((".arff files", "*.arff*"), ("all files", "*.*")))
        if not filename:
            return
        if filename.endswith(".arff"):
            self.path = filename
            self.machine.load_data(filename)
            self.update_info()
            self.draw_plot()
        else:
            messagebox.showerror("Error", "Wrong file format!")

    def on_perceptron(self):
        if not isinstance(self.machine, Machine) or isinstance(self.machine, AdalineMachine):
            self.machine = Machine()
            if self.path is not None: self.machine.load_data(self.path)
            self.update_info()
            messagebox.showinfo("Info", "Switched to Perceptron!")
        else:
            messagebox.showerror("Error", "You are already using Perceptron.")

    def on_adeline(self):
        if not isinstance(self.machine, AdalineMachine):
            self.machine = AdalineMachine()
            if self.path is not None: self.machine.load_data(self.path)
            self.update_info()
            messagebox.showinfo("Info", "Switched to Adaline!")
        else:
            messagebox.showerror("Error", "You are already using Adaline.")

    #side bar callbacks
    def on_learn_steps(self):
        if self.machine.data is None:
            messagebox.showwarning("Warning", "Load a dataset first.")
            return
        steps = self.steps_var.get()
        self.machine.learn_in_steps(steps)
        self.machine.calculate_error()
        self.update_info()
        self.draw_plot()

    def on_learn_slowly(self):
        if self.machine.data is None:
            messagebox.showwarning("Warning", "Load a dataset first.")
            return
        steps = self.steps_var.get()
        self.learn_slowly_step(steps)

    def learn_until(self):
        if self.machine.data is None:
            messagebox.showwarning("Warning", "Load a dataset first.")
            return
        self.learn_until_satep(0)

    def learn_untill(self):
        self.machine.learn_untill_treshold_or_make_ten_thousand_steps()
    
    def on_update_error(self):
        try:
            t = float(self.error_expected.get())
            if t < 1.0 and t > 0.0:
                self.machine.set_tolerance(t)
                self.update_info()
                self.draw_plot()
            else:
                messagebox.showwarning("Warning", "Set en error to a value from 0.0 to 1.0.")
        except Exception as e:
            messagebox.showwarning("Warnning", f"An unexpected error has accured: {e}")

    #step util functions
    def learn_slowly_step(self, remaining):
        if remaining <= 0:
            return
        self.machine.learn()
        self.machine.calculate_error()
        self.update_info()
        self.draw_plot()
        self.root.after(10, self.learn_slowly_step, remaining - 1)

    def learn_until_step(self, steps):
        
        if self.machine.tolerance == 0.0:
            messagebox.showerror("Error", "Treshold has not been set")
            return
        if self.machine.error_scores[-1] < self.machine.tolerance:
            messagebox.showinfo("Sucess", "Treshold has been reached. Yuppi")
            return
        if steps > 9999:
            messagebox.showinfo("Info", "Learning unsucesfull after 9999 steps")
        self.machine.learn()
        self.machine.calculate_error()
        self.update_info()
        self.draw_plot()
        self.root.after(10, self.learn_until_step, steps + 1)

    #plot

    def draw_plot(self):
        self.draw_boundary()
        self.draw_error()
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def draw_boundary(self):
        ax = self.ax
        ax.clear()
        ax.set_title("Decision boundary", fontsize=11)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        m = self.machine

        if m.data is None:
            ax.text(0.5, 0.5, "Load a dataset to see the plot",
                    ha='center', va='center', transform=ax.transAxes,
                    color='grey', fontsize=10)
            return

        x1 = m.data[:, 0].astype(float)
        x2 = m.data[:, 1].astype(float)
        labels = m.class_val[:, 0].astype(int)

        colors = np.where(labels == 1, '#2196F3', '#FF5722')
        ax.scatter(x1, x2, c=colors, s=28, alpha=0.8, linewidths=0,
                   label=None, zorder=3)

        
        ax.legend(handles=[
            mpatches.Patch(color='#2196F3', label='class +1'),
            mpatches.Patch(color='#FF5722', label='class −1 (0)'),
        ], fontsize=8, loc='best')

        w1, w2, bias = m.w1, m.w2, m.bias

        x1_min, x1_max = x1.min(), x1.max()
        margin = (x1_max - x1_min) * 0.1 or 1.0
        xs = np.linspace(x1_min - margin, x1_max + margin, 300)

        if abs(w2) > 1e-9:
            ys = -(w1 * xs + bias) / w2
            ax.plot(xs, ys, color='#4CAF50', linewidth=1.8, label=f'boundary (epoch {m.epoch})', zorder=4)
            ax.legend(fontsize=8)
        else:
            if abs(w1) > 1e-9:
                xv = -bias / w1
                ax.axvline(xv, color='#4CAF50', linewidth=1.8, label=f'boundary x1={xv:.2f}')
                ax.legend(fontsize=8)

        pad_x = (x1_max - x1_min) * 0.12 or 1.0
        pad_y = (x2.max() - x2.min()) * 0.12 or 1.0
        ax.set_xlim(x1_min - pad_x, x1_max + pad_x)
        ax.set_ylim(x2.min() - pad_y, x2.max() + pad_y)

    def draw_error(self):
        ax = self.ax_err
        ax.clear()
        ax.set_title("Learning error", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Error")

        m = self.machine
        scores = m.error_scores if m.error_scores else []

        if not scores:
            ax.text(0.5, 0.5, "No learning history yet",
                    ha='center', va='center', transform=ax.transAxes,
                    color='grey', fontsize=10)
            return

        xs = list(range(1, len(scores) + 1))
        ax.plot(xs, scores, color='#E91E63', linewidth=1.6, zorder=3)
        ax.fill_between(xs, scores, alpha=0.15, color='#E91E63')

        #tolerance
        tol = self.machine.tolerance
        ax.axhline(tol, color='#FF9800', linewidth=1.2, linestyle='--',
                   label=f'target error ({tol:.4f})')
        ax.legend(fontsize=8, loc='upper right')

        ax.set_xlim(1, max(len(scores), 2))
        ax.set_ylim(bottom=0)

    def update_info(self):
        m = self.machine
        kind = "Adaline" if type(self.machine) == AdalineMachine else "Perceptron"
        if m.data is None:
            self.info_text.set(f"Mode: {kind}\nNo data loaded.")
            return
        last_err = m.error_scores[-1] if m.error_scores else float('nan')
        self.info_text.set(
            f"Mode: {kind}\n"
            f"Samples: {m.data.shape[0]}\n"
            f"Epoch: {m.epoch}\n"
            f"Step (t): {m.t}\n"
            f"w1 = {m.w1:.4f}\n"
            f"w2 = {m.w2:.4f}\n"
            f"bias = {m.bias:.4f}\n"
            f"Error: {last_err:.4f}\n"
            f"Error treshold: {m.tolerance:.4f}"
        )