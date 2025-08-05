import json
from pathlib import Path
import cv2
from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.spatial import cKDTree

# ---- Parameters for sliders ----
SLIDER_CONFIGS = [
    {"label": "Canny Lower",     "from_": 50,   "to": 700, "resolution": 10, "init": 120},
    {"label": "Canny Upper",     "from_": 100,  "to": 700, "resolution": 10, "init": 200},
    {"label": "Hough Threshold", "from_": 0,    "to": 300, "resolution": 10, "init": 100},
    {"label": "Min Length (px)", "from_": 10,   "to": 150, "resolution": 10, "init": 100},
    {"label": "Max Gap (px)",    "from_": 0,    "to": 20,  "resolution": 2, "init": 10},
    {"label": "Apply Filter",    "from_": 0,    "to": 2,   "resolution": 1,  "init": 0},
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


def draw_lines(rgb, lines):
    if lines is None or len(lines) == 0:
        return rgb
    print(lines.shape)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    return rgb


class ImageApp:
    def __init__(self, master, image_path, grid_path):
        self.master = master
        master.title("Image Viewer with Sliders")
        master.attributes("-fullscreen", True)

        # Load image in OpenCV format (BGR)
        self.original = smart_downsample(cv2.imread(image_path), 4)
        if self.original is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        self.image_clahe = apply_clahe(cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY))

        # Build the KD-tree
        pixels_database = json.load(grid_path.open("r"))
        self.angle_src_xy = np.array(pixels_database[image_path]["src"])[:, [1, 0]] / 4
        self.angle_dest_xy = np.array(pixels_database[image_path]["dest"])[:, [1, 0]] / 4
        self.angle_vectors_xy = self.angle_dest_xy - self.angle_src_xy
        self.angle_vectors_xy /= np.linalg.norm(self.angle_vectors_xy, axis=1).reshape((-1, 1))
        self.tree = cKDTree(self.angle_src_xy)

        # Create a frame for sliders
        self.sliders = []
        slider_frame = tk.Frame(master)
        slider_frame.pack(side=tk.TOP, fill=tk.X)
        self.slider_vars = []
        for i, cfg in enumerate(SLIDER_CONFIGS):
            var = tk.DoubleVar(value=cfg["init"])
            slider = tk.Scale(slider_frame, variable=var, label=cfg["label"],
                              from_=cfg["from_"], to=cfg["to"], resolution=cfg["resolution"],
                              orient=tk.HORIZONTAL, length=150, command=self.update_images)
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
        canny_low       = int(self.slider_vars[0].get())
        canny_hi        = int(self.slider_vars[1].get())
        hough_threshold = int(self.slider_vars[2].get())
        min_length      = int(self.slider_vars[3].get())
        max_gap         = int(self.slider_vars[4].get())
        do_filter       = int(self.slider_vars[5].get())

        # Image 1: CLAHE
        img1 = cv2.cvtColor(self.image_clahe, cv2.COLOR_GRAY2RGB)

        # Image 2: Canny edges
        edges = cv2.Canny(self.image_clahe, canny_low, canny_hi)
        img2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        lines = cv2.HoughLinesP(
            edges,
            rho=2,
            theta=np.deg2rad(2),
            threshold=hough_threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )
        # import pdb; pdb.set_trace()
        if lines is not None:
            lines = lines.squeeze()
            if len(lines.shape) == 1:
                lines = lines.reshape((1, 4))
        if do_filter == 1:
            lines = self.filter_lines(lines)
        elif do_filter == 2:
            lines = np.hstack([self.angle_src_xy[::80], self.angle_dest_xy[::80]]).astype(int)
        img3 = draw_lines(
            cv2.cvtColor(self.image_clahe, cv2.COLOR_GRAY2RGB),
            lines,
        )

        # Display all three
        self.display_image(img1, self.image_labels[0])
        self.display_image(img2, self.image_labels[1])
        self.display_image(img3, self.image_labels[2])

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

    def filter_lines(self, lines):
        if lines is None:
            return lines
        distances, indices = self.tree.query(lines[:, 0:2], k=1)

        # dx, dy
        vectors = lines[:, 2:4] - lines[:, 0:2]
        vectors = vectors / np.linalg.norm(vectors, axis=1).reshape((-1, 1))

        dot = np.einsum("ij,ij->i", vectors, self.angle_vectors_xy[indices])

        # Alignment
        mask = abs(dot) > np.cos(np.deg2rad(5))
        lines = lines[mask]

        if len(lines.shape) == 1:
            lines = lines.reshape((1, 4))

        return lines


if __name__ == "__main__":
    grid_path = Path("/home/eric/Pictures/test_lines/grid20.json")
    image_path = "/home/eric/Pictures/test_lines/000936-01_000504.JPG"
    root = tk.Tk()
    app = ImageApp(root, image_path, grid_path)
    root.mainloop()
