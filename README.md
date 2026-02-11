# Elective-4
Project for Elective for Image processing

# Requirements.txt
This is where you save the all package dependencies

# Image Filter Batch Processor

A partial activity to pass Elective 4 a simple image processing program using OpenCV.

Drop images into per-filter input folders and run the script to generate processed outputs (Gaussian blur, bloom, vignette, color grading,).

# Description
The project aims to automate an image processing program that uses common filters for photography to make the image more visible or clear. For each filter drag and drop your image into the corresponding folders, run the script once, and the processed result will be written or recorded to a output image directory with the filter-prefixed filenames.

Filters included:

Gaussian blur — softens the image using a 15×15 kernel.

Bloom — adds a glow by blending a heavy blur back into the original.

Vignette — darkens edges using a 2D Gaussian mask.

Grading — simple LUT-based color/contrast tweak (cooler blues, slight lift).

Negative — color invert for a film-negative look.

Each image is resized to 600×400 before filtering for consistency.


# Getting Started
Dependencies
- Installing
  - Clone or download this repository (or copy the script into a new folder).
    (Optional) Create and activate a virtual environment
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
- Install dependencies
  - python -m pip install --upgrade pip
    pip install opencv-python numpy
# Project creates its own folder structure on first run, but you can also create them manually
    input\_gaussian/
    input\_bloom/
    input\_vignette/
    input\_grading/
    input\_negative/
    output\_images/

# Supported Input Formats
.jpg, .jpeg, .png, .bmp, .tiff, .webp



# Executing Program
Place images into any of the input\_<filter>/ folders based on which effect you want:
input\_gaussian/
input\_bloom/
input\_vignette/
input\_grading/
input\_negative/

Check outputs in:
output\_images/


# Folder Structure
├── IMAGEPROCESSING.py
├── input\_gaussian/
├── input\_bloom/
├── input\_vignette/
├── input\_grading/
├── input\_negative/
└── output\_images/

# How It Works
Automatic folders: The script ensures the required input/output directories exist.
Batch loop: For each filter, it iterates over files in the corresponding input\_<filter>/ folder.
Validation: Skips unreadable files and non-supported extensions.
Preprocess: Resizes images to 600×400 (cv2.resize).
Filter application: Executes the matching function.
Save: Writes results to output\_images/ with a filter prefix.


# Authors
**Authors**
Santos, Romel Andrei — **Programmer**
**Romel4B**

Ferrera, Justine Moises L. — **DevOps**
**Supermouse69**

Zari, Benjamin — **QA**
**BenZariThirdie**

Arevalo, Angelo E — **Presenter**
**Gelobyte**








  



