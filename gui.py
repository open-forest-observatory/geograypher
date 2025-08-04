import cv2
from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.spatial import cKDTree

# ---- Parameters for sliders ----
SLIDER_CONFIGS = [
    {"label": "Scale",         "from_": 0.5, "to": 3.0, "resolution": 0.1, "init": 1.0},
    {"label": "Add Constant",  "from_": -100, "to": 100, "resolution": 1,   "init": 0},
    {"label": "Grayscale?",    "from_": 0,    "to": 1,   "resolution": 1,   "init": 0},
    {"label": "Canny Lower",   "from_": 50,    "to": 700, "resolution": 1,   "init": 120},
    {"label": "Canny Upper",   "from_": 100,    "to": 700, "resolution": 1,   "init": 200},
]


def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)


def smart_downsample(img, factor):
    if factor == 1:
        return img
    h, w, _ = img.shape
    down = resize(img, (h // factor, w // factor), anti_aliasing=True, preserve_range=True)
    return down.astype(np.uint8)


class ImageApp:
    def __init__(self, master, image_path):
        self.master = master
        master.title("Image Viewer with Sliders")
        master.attributes("-fullscreen", True)

        # Load image in OpenCV format (BGR)
        self.original = smart_downsample(cv2.imread(image_path), 4)
        if self.original is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        self.image_clahe = apply_clahe(cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY))

        # Build the KD-tree
        # self.tree = cKDTree(pixels_database[image_path])

        # Create a frame for sliders
        self.sliders = []
        slider_frame = tk.Frame(master)
        slider_frame.pack(side=tk.TOP, fill=tk.X)
        self.slider_vars = []
        for i, cfg in enumerate(SLIDER_CONFIGS):
            var = tk.DoubleVar(value=cfg["init"])
            slider = tk.Scale(slider_frame, variable=var, label=cfg["label"],
                              from_=cfg["from_"], to=cfg["to"], resolution=cfg["resolution"],
                              orient=tk.HORIZONTAL, length=200, command=self.update_images)
            slider.pack(side=tk.LEFT)
            self.slider_vars.append(var)

        # Grid layout for images
        self.image_labels = []
        image_frame = tk.Frame(master)
        image_frame.pack()

        # Top row: Image 1 and Image 2
        label1 = tk.Label(image_frame)
        label1.grid(row=0, column=0, padx=5, pady=5)
        label2 = tk.Label(image_frame)
        label2.grid(row=0, column=1, padx=5, pady=5)

        # Bottom row: Image 3 spans both columns
        label3 = tk.Label(image_frame)
        label3.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.image_labels = [label1, label2, label3]

        # Initial update
        self.update_images()

    def update_images(self, event=None):
        # Get values from sliders
        scale     = self.slider_vars[0].get()
        constant  = self.slider_vars[1].get()
        grayscale = self.slider_vars[2].get() > 0
        canny_low = int(self.slider_vars[3].get())
        canny_hi  = int(self.slider_vars[4].get())

        # Image 1: CLAHE
        img1 = cv2.cvtColor(self.image_clahe, cv2.COLOR_GRAY2RGB)

        # Image 2: Canny edges
        edges = cv2.Canny(self.image_clahe, canny_low, canny_hi)
        img2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # distances, indices = self.tree.query(query_pixels, k=1)

        # Display all three
        self.display_image(img1, self.image_labels[0])
        self.display_image(img2, self.image_labels[1])
        self.display_image(img1, self.image_labels[2])

    def display_image(self, img, label_widget):
        # Convert image to PIL Image, then to ImageTk
        img_pil = Image.fromarray(img)
        if label_widget == self.image_labels[2]:
            img_pil = img_pil.resize((int(528 * 1.5), int(395 * 1.5)))
        else:
            img_pil = img_pil.resize((int(528 * 1.0), int(395 * 1.0)))
        tk_img = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=tk_img)
        label_widget.image = tk_img  # Prevent garbage collection

if __name__ == "__main__":
    image_path = "/home/eric/Pictures/test_lines/000936-01_000304.JPG"
    root = tk.Tk()
    app = ImageApp(root, image_path)
    root.mainloop()
