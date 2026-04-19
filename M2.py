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
        self.root.geometry("1000x850")

        self.cv_img = None  
        
        # --- UI Header ---
        controls = tk.Frame(root)
        controls.pack(side="top", pady=10)

        # 1. Step Selection
        tk.Label(controls, text="Main Step:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        self.step_var = ttk.Combobox(controls, values=["Step 1: Row/Col Swap", "Step 2: Pixel Permutation"], width=25)
        self.step_var.current(0)
        self.step_var.grid(row=0, column=1, padx=5)
        self.step_var.bind("<<ComboboxSelected>>", self.toggle_ui)

        # 2. Key Input
        tk.Label(controls, text="Key (Seed):").grid(row=0, column=2, padx=5)
        self.key_entry = tk.Entry(controls, width=8)
        self.key_entry.insert(0, "42")
        self.key_entry.grid(row=0, column=3, padx=5)

        # 3. Dynamic Controls Container
        self.dynamic_frame = tk.Frame(controls)
        self.dynamic_frame.grid(row=0, column=4, padx=10)

        # Step 1 specific UI (Row vs Col)
        self.s1_label = tk.Label(self.dynamic_frame, text="Axis:")
        self.axis_var = ttk.Combobox(self.dynamic_frame, values=["Rows", "Columns"], width=10)
        self.axis_var.current(0)

        # Step 2 specific UI (Manual vs NumPy)
        self.s2_label = tk.Label(self.dynamic_frame, text="Algo:")
        self.algo_var = ttk.Combobox(self.dynamic_frame, values=["Fisher-Yates (Manual)", "NumPy Built-in"], width=20)
        self.algo_var.current(1)

        # 4. Action Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Scramble", command=lambda: self.process(True), bg="#d4edda", width=15).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Unscramble", command=lambda: self.process(False), bg="#f8d7da", width=15).grid(row=0, column=2, padx=5)

        # 5. Function Dashboard (Refined Log)
        self.log_frame = tk.LabelFrame(root, text="Function Dashboard", font=("Arial", 10, "bold"))
        self.log_frame.pack(fill="x", padx=40, pady=10)

        self.math_label = tk.Label(self.log_frame, text="f(x) = None", font=("Courier New", 14, "bold"), fg="#2c3e50")
        self.math_label.pack(pady=5)

        self.detail_label = tk.Label(self.log_frame, text="Waiting for user input...", font=("Arial", 10), fg="#7f8c8d")
        self.detail_label.pack(pady=2)

        self.time_label = tk.Label(root, text="Computation Time: 0.000s", font=("Arial", 9, "italic"))
        self.time_label.pack()

        # 6. Canvas for Image Preview
        self.canvas = tk.Label(root)
        self.canvas.pack(expand=True, pady=10)

        self.toggle_ui()

    def toggle_ui(self, event=None):
        """Dynamic UI switching based on Step selection."""
        self.s1_label.grid_forget()
        self.axis_var.grid_forget()
        self.s2_label.grid_forget()
        self.algo_var.grid_forget()

        if "Step 1" in self.step_var.get():
            self.s1_label.grid(row=0, column=0)
            self.axis_var.grid(row=0, column=1)
        else:
            self.s2_label.grid(row=0, column=0)
            self.algo_var.grid(row=0, column=1)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            self.cv_img = cv2.imread(path)
            self.display_image(self.cv_img)
            self.detail_label.config(text=f"Loaded: {path.split('/')[-1]}")

    def get_permutation(self, size, seed, use_manual):
        """Generates P using Seed. Times the generation process."""
        np.random.seed(seed)
        start = time.perf_counter()
        
        if use_manual:
            p = np.arange(size)
            # Explicit Fisher-Yates
            for i in range(size - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                p[i], p[j] = p[j], p[i]
        else:
            p = np.random.permutation(size)
            
        dur = time.perf_counter() - start
        self.time_label.config(text=f"P Generation Time: {dur:.4f}s")
        return p

    def process(self, is_scramble):
        if self.cv_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            seed = int(self.key_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Key must be a valid integer!")
            return

        h, w, c = self.cv_img.shape
        step_text = self.step_var.get()

        if "Step 1" in step_text:
            axis = self.axis_var.get()
            size = h if axis == "Rows" else w
            p = self.get_permutation(size, seed, False)
            
            if is_scramble:
                # Forward Function P(x)
                self.math_label.config(text=f"P({axis}_i) \u2192 {axis}[P[i]]", fg="#27ae60")
                self.detail_label.config(text=f"Reordered {axis} using permutation vector.")
                if axis == "Rows":
                    self.cv_img = self.cv_img[p, :, :]
                else:
                    self.cv_img = self.cv_img[:, p, :]
            else:
                # Reverse Function P^-1(x)
                p_inv = np.argsort(p)
                self.math_label.config(text=f"P\u207b\u00b9({axis}_j) \u2192 {axis}[P_inv[j]]", fg="#2980b9")
                self.detail_label.config(text=f"Applied Argsort(P) to restore {axis} positions.")
                if axis == "Rows":
                    self.cv_img = self.cv_img[p_inv, :, :]
                else:
                    self.cv_img = self.cv_img[:, p_inv, :]

        else:
            # Step 2: Pixel Permutation
            flat = self.cv_img.reshape(-1, c)
            num_pixels = len(flat)
            algo_choice = "Fisher-Yates (Manual)" in self.algo_var.get()
            
            # Warning for very large images with Manual FY
            if algo_choice and num_pixels > 1000000:
                if not messagebox.askyesno("Performance Warning", "This image has >1M pixels. Fisher-Yates (Manual) will be slow. Continue?"):
                    return

            p = self.get_permutation(num_pixels, seed, algo_choice)
            
            if is_scramble:
                self.math_label.config(text="P(Pixel_i) \u2192 Flat[P[i]]", fg="#27ae60")
                self.detail_label.config(text=f"Full pixel shuffle using {self.algo_var.get()}")
                self.cv_img = flat[p].reshape(h, w, c)
            else:
                # Manual Inversion Logic P^-1
                p_inv = np.empty_like(p)
                p_inv[p] = np.arange(len(p))
                self.math_label.config(text="P\u207b\u00b9(Pixel_j) \u2192 Flat[P_inv[j]]", fg="#2980b9")
                self.detail_label.config(text="Restored pixel distribution from inverse vector.")
                self.cv_img = flat[p_inv].reshape(h, w, c)

        self.display_image(self.cv_img)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # Scale preview to fit screen
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ScramblerApp(root)
    root.mainloop()