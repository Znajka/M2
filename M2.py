import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time

class ScramblerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("M2 - Image Scrambler")
        self.root.geometry("1400x850") # Wide for 3-image layout

        self.img_orig = None  
        self.img_scrambled = None 
        self.img_unscrambled = None

        # --- UI Header ---
        controls = tk.Frame(root)
        controls.pack(side="top", pady=10)

        # Step Selection
        tk.Label(controls, text="Step:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        self.step_var = ttk.Combobox(controls, values=[
            "Step 1: Row/Col Swap", 
            "Step 2: Fisher-Yates Pixel", 
            "Step 3: Class A Substitution"
        ], width=25)
        self.step_var.current(2)
        self.step_var.grid(row=0, column=1, padx=5)
        self.step_var.bind("<<ComboboxSelected>>", self.toggle_ui)

        # Key Input
        tk.Label(controls, text="Key:").grid(row=0, column=2, padx=5)
        self.key_entry = tk.Entry(controls, width=10)
        self.key_entry.insert(0, "42")
        self.key_entry.grid(row=0, column=3, padx=5)

        # Dynamic Controls Container
        self.dynamic_frame = tk.Frame(controls)
        self.dynamic_frame.grid(row=0, column=4, padx=10)

        self.s1_label = tk.Label(self.dynamic_frame, text="Axis:")
        self.axis_var = ttk.Combobox(self.dynamic_frame, values=["Rows", "Columns"], width=10)
        self.axis_var.current(0)

        self.s2_label = tk.Label(self.dynamic_frame, text="Algo:")
        self.algo_var = ttk.Combobox(self.dynamic_frame, values=["Fisher-Yates (Manual)", "NumPy Built-in"], width=20)
        self.algo_var.current(1)

        # Action Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="SCRAMBLE", command=self.run_scramble, bg="#d4edda").grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="UNSCRAMBLE", command=lambda: self.run_unscramble(False), bg="#f8d7da").grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="UNSCRAMBLE (Key-1)", command=lambda: self.run_unscramble(True), bg="#fff3cd").grid(row=0, column=3, padx=5)
        tk.Button(btn_frame, text="Save Result", command=self.save_image).grid(row=0, column=4, padx=5)

        # --- Dashboard ---
        self.log_frame = tk.LabelFrame(root, text="Function Dashboard")
        self.log_frame.pack(fill="x", padx=40, pady=5)
        self.math_label = tk.Label(self.log_frame, text="f(p,k) = ...", font=("Courier", 12, "bold"))
        self.math_label.pack()

        # --- Image Display Area ---
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(pady=10)

        self.canv_orig = self.create_preview(self.display_frame, "Original", 0)
        self.canv_scram = self.create_preview(self.display_frame, "Scrambled", 1)
        self.canv_unscram = self.create_preview(self.display_frame, "Unscrambled", 2)

        self.toggle_ui()

    def create_preview(self, parent, label, col):
        frame = tk.Frame(parent, bd=1, relief="sunken")
        frame.grid(row=0, column=col, padx=10)
        tk.Label(frame, text=label, font=("Arial", 10, "bold")).pack()
        canvas = tk.Label(frame)
        canvas.pack()
        return canvas

    def toggle_ui(self, event=None):
        self.s1_label.grid_forget()
        self.axis_var.grid_forget()
        self.s2_label.grid_forget()
        self.algo_var.grid_forget()
        if "Step 1" in self.step_var.get():
            self.s1_label.grid(row=0, column=0); self.axis_var.grid(row=0, column=1)
        elif "Step 2" in self.step_var.get():
            self.s2_label.grid(row=0, column=0); self.algo_var.grid(row=0, column=1)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_orig = cv2.imread(path)
            self.update_display(self.img_orig, self.canv_orig)

    def get_p(self, size, seed, manual):
        np.random.seed(seed)
        if manual:
            p = np.arange(size)
            for i in range(size - 1, 0, -1):
                j = np.random.randint(0, i+1); p[i], p[j] = p[j], p[i]
            return p
        return np.random.permutation(size)

    def run_scramble(self):
        if self.img_orig is None: return
        k = int(self.key_entry.get())
        step = self.step_var.get()
        h, w, c = self.img_orig.shape

        if "Step 1" in step:
            ax = self.axis_var.get()
            p = self.get_p(h if ax=="Rows" else w, k, False)
            self.img_scrambled = self.img_orig[p, :] if ax=="Rows" else self.img_orig[:, p]
            self.math_label.config(text=f"f(p,k): p \u2192 p[Permutation(k)]")
        
        elif "Step 2" in step:
            flat = self.img_orig.reshape(-1, c)
            p = self.get_p(len(flat), k, "Manual" in self.algo_var.get())
            self.img_scrambled = flat[p].reshape(h, w, c)
            self.math_label.config(text=f"f(p,k): p_pos \u2192 FY_Shuffle(p_pos, k)")

        elif "Step 3" in step:
            self.img_scrambled = ((self.img_orig.astype(np.int16) + k) % 256).astype(np.uint8)
            self.math_label.config(text=f"f(p,k): (p + k) mod 256")

        self.update_display(self.img_scrambled, self.canv_scram)

    def run_unscramble(self, wrong_key):
        if self.img_scrambled is None: return
        k = int(self.key_entry.get())
        if wrong_key: k -= 1
        step = self.step_var.get()
        h, w, c = self.img_scrambled.shape

        if "Step 1" in step:
            ax = self.axis_var.get()
            p = self.get_p(h if ax=="Rows" else w, k, False)
            p_inv = np.argsort(p)
            self.img_unscrambled = self.img_scrambled[p_inv, :] if ax=="Rows" else self.img_scrambled[:, p_inv]
            self.math_label.config(text=f"f\u207b\u00b9(p,k): p[Argsort(Permutation(k))]")

        elif "Step 2" in step:
            flat = self.img_scrambled.reshape(-1, c)
            p = self.get_p(len(flat), k, "Manual" in self.algo_var.get())
            p_inv = np.empty_like(p); p_inv[p] = np.arange(len(p))
            self.img_unscrambled = flat[p_inv].reshape(h, w, c)
            self.math_label.config(text=f"f\u207b\u00b9(p,k): Inverse_Mapping(p_pos, k)")

        elif "Step 3" in step:
            self.img_unscrambled = ((self.img_scrambled.astype(np.int16) - k) % 256).astype(np.uint8)
            self.math_label.config(text=f"f\u207b\u00b9(p,k): (p - k) mod 256")

        self.update_display(self.img_unscrambled, self.canv_unscram)

    def update_display(self, cv_img, canvas):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400)) # Scale down to fit 3 in a row
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.config(image=img_tk)
        canvas.image = img_tk

    def save_image(self):
        if self.img_unscrambled is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path: cv2.imwrite(path, self.img_unscrambled)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScramblerApp(root)
    root.mainloop()