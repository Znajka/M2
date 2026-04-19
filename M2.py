import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ScramblerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("M2 - Image Scrambler")
        self.root.geometry("1350x850") 

        # Persistent Memory States
        self.img_orig = None    
        self.img_work = None    # This is the "Working Canvas" that remembers every change

        # --- UI Header ---
        controls = tk.Frame(root)
        controls.pack(side="top", pady=10)

        self.step_var = ttk.Combobox(controls, values=[
            "Step 1: Row/Col Swap", 
            "Step 2: Fisher-Yates Pixel", 
            "Step 3: Class A Substitution"
        ], width=25)
        self.step_var.current(0)
        self.step_var.grid(row=0, column=1, padx=5)
        self.step_var.bind("<<ComboboxSelected>>", self.toggle_ui)

        tk.Label(controls, text="Key (k):").grid(row=0, column=2, padx=5)
        self.key_entry = tk.Entry(controls, width=8)
        self.key_entry.insert(0, "150")
        self.key_entry.grid(row=0, column=3, padx=5)

        self.dynamic_frame = tk.Frame(controls)
        self.dynamic_frame.grid(row=0, column=4, padx=5)
        self.s1_ui = ttk.Combobox(self.dynamic_frame, values=["Rows", "Columns"], width=10)
        self.s1_ui.current(0)
        self.s2_ui = ttk.Combobox(self.dynamic_frame, values=["Fisher-Yates (Manual)", "NumPy Built-in"], width=20)
        self.s2_ui.current(1)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=12).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="SCRAMBLE", command=self.run_scramble, bg="#d4edda", width=12).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="UNSCRAMBLE", command=lambda: self.run_unscramble(False), bg="#f8d7da", width=12).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="UNSCRAMBLE (Key-1)", command=lambda: self.run_unscramble(True), bg="#fff3cd", width=18).grid(row=0, column=3, padx=5)
        tk.Button(btn_frame, text="Save Memory", command=self.save_results, width=12).grid(row=0, column=4, padx=5)

        self.log_frame = tk.LabelFrame(root, text="Algorithm Log")
        self.log_frame.pack(fill="x", padx=40, pady=5)
        self.math_label = tk.Label(self.log_frame, text="", font=("Courier New", 11, "bold"), fg="#2c3e50")
        self.math_label.pack(pady=10)

        # --- 3 Preview Layout ---
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(pady=10)
        self.canv_orig = self.create_preview(self.display_frame, "1. Original", 0)
        self.canv_scram = self.create_preview(self.display_frame, "2. Last Scramble State", 1)
        self.canv_unscram = self.create_preview(self.display_frame, "3. Last Unscramble State", 2)

        self.toggle_ui()

    def create_preview(self, parent, label, col):
        frame = tk.Frame(parent, bd=1, relief="sunken")
        frame.grid(row=0, column=col, padx=15)
        tk.Label(frame, text=label, font=("Arial", 10, "bold")).pack()
        canvas = tk.Label(frame); canvas.pack()
        return canvas

    def toggle_ui(self, event=None):
        self.s1_ui.grid_forget(); self.s2_ui.grid_forget()
        step = self.step_var.get()
        if "Step 1" in step: self.s1_ui.grid(row=0, column=0)
        elif "Step 2" in step: self.s2_ui.grid(row=0, column=0)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_orig = cv2.imread(path)
            self.img_work = self.img_orig.copy() 
            self.update_display(self.img_orig, self.canv_orig)
            # Reset visual previews on new load
            self.canv_scram.config(image='')
            self.canv_unscram.config(image='')
            self.math_label.config(text="")

    def get_p(self, size, seed, manual):
        np.random.seed(seed)
        p = np.arange(size)
        if manual:
            for i in range(size - 1, 0, -1):
                j = np.random.randint(0, i+1); p[i], p[j] = p[j], p[i]
        else: p = np.random.permutation(size)
        return p

    def run_scramble(self):
        if self.img_work is None: return
        
        k = int(self.key_entry.get())
        step = self.step_var.get()
        h, w, c = self.img_work.shape

        if "Step 1" in step:
            ax = self.s1_ui.get()
            p = self.get_p(h if ax=="Rows" else w, k, False)
            self.img_work = self.img_work[p, :] if ax=="Rows" else self.img_work[:, p]
            self.math_label.config(text=f"Step 1: Scrambled {ax}. Logic: f(p,k) = p[P(k)]")
        
        elif "Step 2" in step:
            flat = self.img_work.reshape(-1, c)
            p = self.get_p(len(flat), k, "Manual" in self.s2_ui.get())
            self.img_work = flat[p].reshape(h, w, c)
            self.math_label.config(text=f"Step 2: Scrambled Pixels. Logic: f(p,k) = FY_Shuffle(p, k)")

        elif "Step 3" in step:
            self.img_work = ((self.img_work.astype(np.int16) + k) % 256).astype(np.uint8)
            self.math_label.config(text=f"Step 3: Applied Substitution. Logic: f(p,k) = (p + {k}) mod 256")

        self.update_display(self.img_work, self.canv_scram)

    def run_unscramble(self, wrong_key):
        if self.img_work is None: return
        
        k_val = int(self.key_entry.get())
        k = k_val - 1 if wrong_key else k_val
        step = self.step_var.get()
        h, w, c = self.img_work.shape

        if "Step 3" in step:
            self.img_work = ((self.img_work.astype(np.int16) - k) % 256).astype(np.uint8)
            self.math_label.config(text=f"Step 3 (Inv): Reversed Substitution. Logic: f\u207b\u00b9(p,k) = (p - {k}) mod 256")

        elif "Step 1" in step:
            ax = self.s1_ui.get()
            p = self.get_p(h if ax=="Rows" else w, k, False)
            p_inv = np.argsort(p)
            self.img_work = self.img_work[p_inv, :] if ax=="Rows" else self.img_work[:, p_inv]
            self.math_label.config(text=f"Step 1 (Inv): Reversed {ax} Swap. Logic: f\u207b\u00b9(p,k) = p[Argsort(P(k))]")

        elif "Step 2" in step:
            flat = self.img_work.reshape(-1, c)
            p = self.get_p(len(flat), k, "Manual" in self.s2_ui.get())
            p_inv = np.empty_like(p); p_inv[p] = np.arange(len(p))
            self.img_work = flat[p_inv].reshape(h, w, c)
            self.math_label.config(text=f"Step 2 (Inv): Reversed Pixel Shuffle. Logic: f\u207b\u00b9(p,k) = p[P_inv(k)]")

        self.update_display(self.img_work, self.canv_unscram)

    def save_results(self):
        if self.img_work is not None:
            path = filedialog.asksaveasfilename(title="Save Current State", defaultextension=".png")
            if path: cv2.imwrite(path, self.img_work)

    def update_display(self, cv_img, canvas):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.config(image=img_tk)
        canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ScramblerApp(root)
    root.mainloop()