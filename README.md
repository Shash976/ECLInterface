# Electrochemiluminescence Interface

Desktop application for end-to-end ECL biosensor analysis — from image capture to ML-based concentration prediction.

Built at the [MEMS, Microfluidics and Nanoelectronics Lab](http://www.mmne.in), BITS-Pilani Hyderabad Campus.

Code for https://doi.org/10.1016/j.compbiomed.2024.109546

---

## What it does

1. **Image Analysis** — Batch-processes ECL sensor images (JPG/PNG/GIF). Extracts light intensity by masking pixels in the reagent's HSV hue range and computing the mean of the region of interest. Exports intensity–concentration pairs to Excel.
2. **Data Analysis** — Trains multiple regression models (Linear, RANSAC, Huber, Theil-Sen, Decision Tree, Random Forest, AdaBoost, Gradient Boost, KNN, SVM) on the exported data. Saves trained models as `.pkl` files and generates scatter plots + error metrics (R², MAE, RMSE).
3. **Prediction** — Loads trained models and predicts concentration from either a new image or a manually entered intensity value.
4. **Calibration** — Interactive HSV/LAB color-space masking tool with live sliders. Lets you visually tune a reagent's hue range and apply it directly to the analysis pipeline.
5. **Real-Time** *(Raspberry Pi only)* — Live ECL analysis from PiCamera v2.

Supported reagents: **Luminol** (hue 110–130) and **Ruthenium** (hue 0–20). Auto-detection is available for both.

---

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.9+ |
| OS | Windows / macOS / Linux |
| RAM | 4 GB |
| For real-time tab | Raspberry Pi 3+ with PiCamera v2+ |

---

## Setup

```bash
git clone https://github.com/Shash976/ECLInterface.git
cd ECLInterface
pip install -r requirements.txt
python src/main.py
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover the full scientific pipeline (image processing, ML training, prediction) and run headlessly — no display or camera required.

---

## Project structure

```
ECLInterface/
├── src/
│   ├── main.py               # Entry point
│   ├── ml_gui_pyqt5.py       # Main window and all tabs
│   ├── image_processor.py    # Batch-processing state and timer (ImageProcessor)
│   ├── image_analysis.py     # HSV masking, intensity extraction
│   ├── processing.py         # ML training pipeline, Excel export
│   ├── prediction.py         # Model loading and inference
│   ├── calibration.py        # Interactive color-space calibration tool
│   ├── cameraApp.py          # Real-time PiCamera tab
│   ├── model_def.py          # ML model registry, Reagent definitions, DataAxis
│   ├── util.py               # Shared utilities (crop, GIF, platform open)
│   └── media/                # Images used by the UI
├── tests/
│   ├── conftest.py           # Qt mock, matplotlib backend, sys.path setup
│   ├── test_util.py
│   ├── test_model_def.py
│   ├── test_image_analysis.py
│   ├── test_processing.py
│   └── test_prediction.py
└── requirements.txt
```

---

## Image folder convention

The batch image analysis tab expects a folder laid out as:

```
experiment_folder/
├── 0.1 uM/
│   ├── image1.jpg
│   └── image2.jpg
├── 0.5 uM/
│   └── image1.jpg
└── 1.0 uM/
    └── image1.gif
```

Each subfolder name must contain a numeric value — that value becomes the concentration label. Subfolders are sorted numerically before processing. GIF files are handled by taking the max-intensity frame across all frames.

---

## Calibrating a reagent

1. Open the **Calibrate** tab and browse to a representative image.
2. Switch to **HSV** mode.
3. Adjust the six sliders (lower H/S/V and upper H/S/V) until only the emission region is visible in the mask panel.
4. Select the target reagent from the dropdown and click **Apply to Reagent**. The hue range is applied to the analysis pipeline immediately for the current session.
