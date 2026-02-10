import os
import cv2
import numpy as np

SUPPORTED_INPUTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

FILTERS = {
    "gaussian": {"input": "input_gaussian", "output": "output_images"},
    "bloom": {"input": "input_bloom", "output": "output_images"},
    "vignette": {"input": "input_vignette", "output": "output_images"},
    "grading": {"input": "input_grading", "output": "output_images"},
    "flare": {"input": "input_flare", "output": "output_images"},
    "cartoon": {"input": "input_cartoon", "output": "output_images"},
    "sepia": {"input": "input_sepia", "output": "output_images"},
    "rgb_glitch": {"input": "input_rgb_glitch", "output": "output_images"},
    "neon_glow": {"input": "input_neon_glow", "output": "output_images"},
    "vhs_filter": {"input": "input_vhs_filter", "output": "output_images"},
}

for f in FILTERS.values():
    os.makedirs(f["input"], exist_ok=True)
    os.makedirs(f["output"], exist_ok=True)

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

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 2
    )

    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def sepia(img):
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    out = cv2.transform(img, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def rgb_glitch(img):
    b, g, r = cv2.split(img)
    rows, cols = img.shape[:2]

    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=0)

    return cv2.merge((b, g, r))

def neon_glow(img):
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges, None)
    edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    return cv2.addWeighted(img, 0.8, edges_col, 0.6, 0)

def vhs_filter(img):
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    lines = img.copy()
    for i in range(0, img.shape[0], 4):
        lines[i:i+1, :] = 0

    return cv2.addWeighted(img, 0.9, lines, 0.1, 0)



FILTER_FUNCS = {
    "gaussian": gaussian,
    "bloom": bloom,
    "vignette": vignette,
    "grading": grading,
    "flare": flare,
    "cartoon": cartoon,
    "sepia": sepia,
    "rgb_glitch": rgb_glitch,
    "neon_glow": neon_glow,
    "vhs_filter": vhs_filter,
}

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
