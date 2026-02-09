import os
import time
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

RUN_MODE = os.getenv("RUN_MODE", "local")  # local | github

SUPPORTED_INPUTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

FILTERS = {
    "gaussian": {"input": "input_gaussian", "output": "output_gaussian"},
    "bloom": {"input": "input_bloom", "output": "output_bloom"},
    "vignette": {"input": "input_vignette", "output": "output_vignette"},
    "grading": {"input": "input_grading", "output": "output_grading"},
    "flare": {"input": "input_flare", "output": "output_flare"},
}

for f in FILTERS.values():
    os.makedirs(f["input"], exist_ok=True)
    os.makedirs(f["output"], exist_ok=True)

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
        lut[i, 0] = [
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

FILTER_FUNCTIONS = {
    "gaussian": apply_gaussian_blur,
    "bloom": apply_bloom,
    "vignette": apply_vignette,
    "grading": apply_color_grading,
    "flare": apply_lens_flare,
}


def process_image(image_path, filter_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load {image_path}")
        return

    img = cv2.resize(img, (600, 400))

    name, ext = os.path.splitext(os.path.basename(image_path))
    output_dir = FILTERS[filter_name]["output"]
    output_path = os.path.join(output_dir, f"{name}_{filter_name}{ext}")

    processed = FILTER_FUNCTIONS[filter_name](img)
    cv2.imwrite(output_path, processed)

    print(f"✅ {filter_name} applied → {output_path}")


class FilterHandler(FileSystemEventHandler):
    def __init__(self, filter_name):
        self.filter_name = filter_name

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(SUPPORTED_INPUTS):
            time.sleep(1)  # ensure file fully saved
            process_image(event.src_path, self.filter_name)

def github_test_run():
    print("🐙 Running in GitHub mode")

    test_image = "test.jpg"
    if not os.path.exists(test_image):
        print("⚠️ test.jpg not found — skipping")
        return

    img = cv2.imread(test_image)
    if img is None:
        print("❌ Failed to load test image")
        return

    img = cv2.resize(img, (600, 400))
    os.makedirs("github_outputs", exist_ok=True)

    for name, func in FILTER_FUNCTIONS.items():
        out = func(img)
        cv2.imwrite(f"github_outputs/{name}.jpg", out)
        print(f"✅ {name} filter generated")

    print("🎉 GitHub test completed")

if RUN_MODE == "github":
    github_test_run()
else:
    observers = []

    for filter_name, paths in FILTERS.items():
        handler = FilterHandler(filter_name)
        observer = Observer()
        observer.schedule(handler, paths["input"], recursive=False)
        observer.start()
        observers.append(observer)
        print(f"👀 Watching {paths['input']}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopping watchers...")
        for o in observers:
            o.stop()
        for o in observers:
            o.join()
