import os
import cv2
import numpy as np

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

# ================= FILTERS =================

def gaussian(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def bloom(img):
    blur = cv2.GaussianBlur(img, (0, 0), 15)
    return cv2.addWeighted(img, 1.2, blur, 0.6, 0)

def vignette(img):
    rows, cols = img.shape[:2]

    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max()

    # Convert image to float
    out = img.astype(np.float32)

    for i in range(3):
        out[:, :, i] = out[:, :, i] * mask

    # Convert back to uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def grading(img):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        lut[i][0] = [
            min(255, int(i * 1.05)),
            min(255, int(i * 1.00)),
            min(255, int(i * 1.15))
        ]
    return cv2.LUT(img, lut)

def flare(img):
    overlay = img.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    for r in range(40, 200, 40):
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

FILTER_FUNCS = {
    "gaussian": gaussian,
    "bloom": bloom,
    "vignette": vignette,
    "grading": grading,
    "flare": flare,
}

# ================= PROCESS =================

def process_all():
    for name, paths in FILTERS.items():
        for file in os.listdir(paths["input"]):
            if file.lower().endswith(SUPPORTED_INPUTS):
                src = os.path.join(paths["input"], file)
                dst = os.path.join(paths["output"], f"{name}_{file}")

                img = cv2.imread(src)
                if img is None:
                    continue

                img = cv2.resize(img, (600, 400))
                out = FILTER_FUNCS[name](img)
                cv2.imwrite(dst, out)
                print(f"✅ {name} → {file}")

if __name__ == "__main__":
    process_all()
    print("🎉 Image processing complete")
