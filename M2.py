import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ScramblerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scrambler App")
        self.root.geometry("900x750")

        self.cv_img = None  
        
        # --- UI Setup ---
        controls = tk.Frame(root)
        controls.pack(side="top", pady=10)

        tk.Label(controls, text="Step:").grid(row=0, column=0)
        self.step_var = ttk.Combobox(controls, values=["Step 1: Row Swap", "Step 2: Fisher-Yates Pixel Permutation"])
        self.step_var.current(1) 
        self.step_var.grid(row=0, column=1, padx=5)

        tk.Label(controls, text="Key:").grid(row=0, column=2)
        self.key_entry = tk.Entry(controls, width=8)
        self.key_entry.insert(0, "42")
        self.key_entry.grid(row=0, column=3, padx=5)

        tk.Button(controls, text="Load Image", command=self.load_image).grid(row=0, column=4, padx=5)
        tk.Button(controls, text="Scramble", command=lambda: self.process(True)).grid(row=0, column=5, padx=5)
        tk.Button(controls, text="Unscramble", command=lambda: self.process(False)).grid(row=0, column=6, padx=5)

        self.func_label = tk.Label(root, text="Function: None", font=("Courier", 10), fg="blue", bg="#f0f0f0", relief="sunken", pady=10)
        self.func_label.pack(fill="x", padx=20, pady=5)

        self.canvas = tk.Label(root)
        self.canvas.pack(expand=True)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.cv_img = cv2.imread(path)
            self.display_image(self.cv_img)

    def get_fisher_yates_permutation(self, size, seed):
        """Explicit Fisher-Yates (Durstenfeld) Shuffle Algorithm"""
        p = np.arange(size)
        np.random.seed(seed)
        for i in range(size - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            p[i], p[j] = p[j], p[i]
        return p

    def get_inverse_permutation(self, p_vector):
        """Inversion Algorithm: P^-1"""
        p_inv = np.empty_like(p_vector)
        p_inv[p_vector] = np.arange(len(p_vector))
        return p_inv

    def process(self, is_scramble):
        if self.cv_img is None: return
        
        try:
            seed = int(self.key_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Key must be an integer")
            return

        h, w, c = self.cv_img.shape
        step = self.step_var.get()

        if "Step 1" in step:
            indices = self.get_fisher_yates_permutation(h, seed)
            if is_scramble:
                self.func_label.config(text=f"P(row): Fisher-Yates swap sequence | Key: {seed}")
                self.cv_img = self.cv_img[indices, :]
            else:
                inv = self.get_inverse_permutation(indices)
                self.func_label.config(text=f"P^-1(row): Inverse Fisher-Yates mapping | Key: {seed}")
                self.cv_img = self.cv_img[inv, :]

        else:
            # Step 2: Pixel Permutation using Fisher-Yates
            flat_img = self.cv_img.reshape(-1, c)
            num_pixels = flat_img.shape[0]
            
            # Show progress if image is large
            self.func_label.config(text="Calculating Fisher-Yates Permutation... Please wait.")
            self.root.update()

            p_vector = self.get_fisher_yates_permutation(num_pixels, seed)
            
            if is_scramble:
                self.func_label.config(text=f"P(px): Fisher-Yates Shuffle | Function: swap(p[i], p[rand(0,i)])")
                self.cv_img = flat_img[p_vector].reshape(h, w, c)
            else:
                p_inv = self.get_inverse_permutation(p_vector)
                self.func_label.config(text=f"P^-1(px): Inverse Algorithm | Function: p_inv[p[i]] = i")
                self.cv_img = flat_img[p_inv].reshape(h, w, c)

        self.display_image(self.cv_img)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((750, 550))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ScramblerApp(root)
    root.mainloop()