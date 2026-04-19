import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class ScramblerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Step 1: Row Scrambler")
        self.root.geometry("800x600")

        self.img_path = None
        self.cv_img = None  # Original Image
        self.mod_img = None # Processed Image

        # --- UI Setup ---
        controls = tk.Frame(root)
        controls.pack(side="top", pady=10)

        tk.Label(controls, text="Key (Integer):").grid(row=0, column=0)
        self.key_entry = tk.Entry(controls)
        self.key_entry.insert(0, "123") # Default key
        self.key_entry.grid(row=0, column=1, padx=5)

        tk.Button(controls, text="Load Image", command=self.load_image).grid(row=0, column=2, padx=5)
        tk.Button(controls, text="Scramble", command=lambda: self.process(True)).grid(row=0, column=3, padx=5)
        tk.Button(controls, text="Unscramble", command=lambda: self.process(False)).grid(row=0, column=4, padx=5)

        # Image Display Area
        self.canvas = tk.Label(root)
        self.canvas.pack(expand=True)

    def load_image(self):
        self.img_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if self.img_path:
            self.cv_img = cv2.imread(self.img_path)
            self.display_image(self.cv_img)

    def process(self, is_scramble):
        if self.cv_img is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        
        try:
            seed = int(self.key_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Key must be an integer!")
            return

        h, w, _ = self.cv_img.shape
        
        # 1. Create a deterministic list of indices
        indices = np.arange(h)
        np.random.seed(seed)
        np.random.shuffle(indices)

        if is_scramble:
            # Reorder rows based on shuffled indices
            self.mod_img = self.cv_img[indices, :]
        else:
            # To unscramble, we find where the original indices went
            # np.argsort returns the inverse of the permutation
            inverse_indices = np.argsort(indices)
            self.mod_img = self.cv_img[inverse_indices, :]
        
        # Overwrite cv_img so we can chain operations or toggle back
        self.cv_img = self.mod_img.copy()
        self.display_image(self.cv_img)

    def display_image(self, img):
        # Convert BGR (OpenCV) to RGB (Tkinter)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for preview if too large
        img_pil.thumbnail((700, 500))
        
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ScramblerApp(root)
    root.mainloop()
