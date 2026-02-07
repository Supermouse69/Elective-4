import os
import time
import threading
import cv2
import numpy as np
from tkinter import *
from tkinter import scrolledtext
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
SUPPORTED_INPUTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

observer = None



def apply_gaussian_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)


def apply_bloom(img):
    blur = cv2.GaussianBlur(img, (0, 0), 15)
    return cv2.addWeighted(img, 1.2, blur, 0.6, 0)


def apply_vignette(img):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    mask = kernel_y * kernel_x.T
    mask /= mask.max()

    output = img.copy()
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask
    return output


def apply_color_grading(img):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        lut[i][0] = [
            np.clip(i * 1.05, 0, 255),
            np.clip(i * 1.00, 0, 255),
            np.clip(i * 1.15, 0, 255)
        ]
    return cv2.LUT(img, lut)


def apply_lens_flare(img):
    overlay = img.copy()
    h, w = img.shape[:2]
    cx = np.random.randint(w // 4, w * 3 // 4)
    cy = np.random.randint(h // 4, h * 3 // 4)

    for r in range(40, 200, 40):
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), -1)

    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)



def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        log_box.insert(END, "❌ Failed to load image\n")
        return

    image = cv2.resize(image, (600, 400))
    name = os.path.splitext(os.path.basename(image_path))[0]
    ext = output_format.get()

    def save(img, suffix):
        cv2.imwrite(f"{OUTPUT_DIR}/{name}_{suffix}.{ext}", img)

    if blur_var.get():
        save(apply_gaussian_blur(image), "gaussian")

    if bloom_var.get():
        save(apply_bloom(image), "bloom")

    if vignette_var.get():
        save(apply_vignette(image), "vignette")

    if grading_var.get():
        save(apply_color_grading(image), "grading")

    if flare_var.get():
        save(apply_lens_flare(image), "flare")

    log_box.insert(END, f"✅ Processed: {name}\n")
    log_box.see(END)



class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_INPUTS):
            time.sleep(1)
            process_image(event.src_path)


def start_watching():
    global observer
    observer = Observer()
    observer.schedule(ImageHandler(), INPUT_DIR, recursive=False)
    observer.start()
    log_box.insert(END, "👀 Folder monitoring started\n")


def stop_watching():
    global observer
    if observer:
        observer.stop()
        observer.join()
        observer = None
        log_box.insert(END, "🛑 Folder monitoring stopped\n")



def drop_image(event):
    files = root.tk.splitlist(event.data)
    for file in files:
        if file.lower().endswith(SUPPORTED_INPUTS):
            process_image(file)
        else:
            log_box.insert(END, "❌ Unsupported file type\n")



root = TkinterDnD.Tk()
root.title("Elective 4")
root.geometry("850x700")
root.resizable(False, False)


bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((850, 700), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


title = Label(root, text="📸 Image Auto Processor",
              font=("Arial", 20, "bold"), bg="#ffffff")
title.pack(pady=10)
title.lift()


drop_label = Label(
    root,
    text="📂 Drag & Drop Images Here",
    font=("Arial", 14, "bold"),
    relief="ridge",
    width=45,
    height=4,
    bg="#f0f0f0"
)
drop_label.pack(pady=10)
drop_label.lift()
drop_label.drop_target_register(DND_FILES)
drop_label.dnd_bind("<<Drop>>", drop_image)


filter_frame = LabelFrame(root, text="Filters", padx=15, pady=10, bg="#ffffff")
filter_frame.pack(pady=5)
filter_frame.lift()

blur_var = BooleanVar()
bloom_var = BooleanVar()
vignette_var = BooleanVar()
grading_var = BooleanVar()
flare_var = BooleanVar()

Checkbutton(filter_frame, text="Gaussian Blur", variable=blur_var, bg="#ffffff").grid(row=0, column=0, sticky="w")
Checkbutton(filter_frame, text="Bloom Effect", variable=bloom_var, bg="#ffffff").grid(row=0, column=1, sticky="w")
Checkbutton(filter_frame, text="Vignette", variable=vignette_var, bg="#ffffff").grid(row=1, column=0, sticky="w")
Checkbutton(filter_frame, text="Color Grading", variable=grading_var, bg="#ffffff").grid(row=1, column=1, sticky="w")
Checkbutton(filter_frame, text="Lens Flare", variable=flare_var, bg="#ffffff").grid(row=2, column=0, sticky="w")


format_frame = LabelFrame(root, text="Output Format", padx=10, pady=5, bg="#ffffff")
format_frame.pack(pady=5)
format_frame.lift()

output_format = StringVar(value="jpg")
OptionMenu(format_frame, output_format, "jpg", "png", "bmp", "tiff", "webp").pack()


Button(root, text="▶ Start Folder Monitoring", width=30,
       command=lambda: threading.Thread(target=start_watching, daemon=True).start()).pack(pady=5)
Button(root, text="⏹ Stop Folder Monitoring", width=30,
       command=stop_watching).pack(pady=5)


Label(root, text="Status Log", font=("Arial", 12), bg="#ffffff").pack()
log_box = scrolledtext.ScrolledText(root, width=100, height=12)
log_box.pack(padx=10, pady=5)
log_box.lift()

Label(root, text="Drag images OR drop them into 'input_images/'",
      font=("Arial", 10, "italic"), bg="#ffffff").pack(pady=5)

root.mainloop()
