# CAPE - Camera-based Attendance Presence Estimator

A desk attendance detection system using ROI-based image classification. Determines whether students are sitting at their desks using fixed camera feeds.

## Overview

This project uses a MobileNetV2 classifier to detect desk occupancy. Unlike traditional object detection, it employs ROI (Region of Interest) based classification:

1. Fixed cameras capture room images
2. Predefined desk regions are drawn on reference images
3. Each crop is classified as `empty` or `occupied`
4. Results show attendance status for each desk

## Project Structure

```
cape/
├── quick_label.py       # ROI labeling tool (draw desk bounding boxes)
├── auto_crop.py        # Crop desk regions from camera images
├── train_classifier.py # Train the occupancy classifier
├── detect.py           # Run detection on new images
├── zones.json          # Desk zone coordinates per camera angle
├── models/             # Trained model checkpoints (created after training)
├── dataset/
│   ├── raw/            # Original camera images
│   ├── sorted/         # Images grouped by camera angle
│   │   ├── angle_1/
│   │   ├── angle_2/
│   │   └── ...
│   ├── crops/          # Cropped desk regions
│   │   ├── desk_a/
│   │   │   ├── to_sort/     # Unlabelled crops
│   │   │   ├── empty/       # User-labeled empty images
│   │   │   └── occupied/    # User-labeled occupied images
│   │   └── ...
│   └── reference/      # Reference images for zone labeling
└── requirements.txt   # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

This is a **human-in-the-loop** system. Each step requires user interaction.

### Step 1: Prepare Images

Manually clean and group images into 6 camera angle folders:

```
dataset/sorted/
├── angle_1/
├── angle_2/
├── angle_3/
├── angle_4/
├── angle_5/
└── angle_6/
```

### Step 2: Label Desk Zones (ROI)

Draw bounding boxes for each desk on a reference image:

```bash
python quick_label.py --angle angle_1
```

**Instructions:**

1. An image window opens with the first image from `dataset/sorted/angle_1/`
2. Draw exactly 3 rectangles in order:
   - **First rectangle** = desk_a
   - **Second rectangle** = desk_b
   - **Third rectangle** = desk_c
3. **Left-click & drag** to draw
4. **Right-click** to undo last rectangle
5. Press **ENTER** to save, **ESC** to cancel

Coordinates are saved to `zones.json`. Repeat for each angle:

```bash
python quick_label.py --angle angle_2
python quick_label.py --angle angle_3
# ... etc
```

### Step 3: Crop Desk Regions

Crop desk regions from all images in an angle folder:

```bash
# Crop only angle_1
python auto_crop.py --angle angle_1

# Crop all angles
python auto_crop.py --all

# Overwrite existing crops
python auto_crop.py --angle angle_1 --force
```

Crops are saved to:

```
dataset/crops/desk_a/to_sort/
dataset/crops/desk_b/to_sort/
dataset/crops/desk_c/to_sort/
```

### Step 4: Label Crops Manually

**IMPORTANT:** This step is done MANUALLY by the user.

The system does NOT automate class labeling. You must visually inspect and move images:

1. Open `dataset/crops/{desk}/to_sort/`
2. Look at each image
3. Move to `empty/` if no one is sitting
4. Move to `occupied/` if someone is sitting

```
dataset/crops/desk_a/
├── to_sort/     <- Generated crops (to be labeled)
├── empty/       <- User moves empty desk images here
└── occupied/    <- User moves occupied desk images here
```

### Step 5: Train Classifier

Train the occupancy classifier:

```bash
# Default settings (20 epochs, batch size 32)
python train_classifier.py

# Custom settings
python train_classifier.py --epochs 30 --batch-size 16 --lr 0.0005
```

Model saved to `models/best_model.pth`

### Step 6: Run Detection

Detect occupancy on new images:

```bash
# Single image
python detect.py --image photo.jpg

# Specify angle (recommended)
python detect.py --image photo.jpg --angle angle_1

# Folder of images
python detect.py --folder dataset/sorted/angle_1

# Indonesian output
python detect.py --image photo.jpg --indo

# Custom threshold
python detect.py --image photo.jpg --threshold 0.7

# Save results to JSON
python detect.py --image photo.jpg --output result.json
```

Example output:

```
desk_a (Asep) → HADIR (0.85)
desk_b (Budi) → TIDAK HADIR (0.54)
desk_c (Siti) → HADIR (0.85)
```

## Configuration

### zones.json

Edit this file to customize:

```json
{
  "desks": ["desk_a", "desk_b", "desk_c"],
  "person_map": {
    "desk_a": "Asep",
    "desk_b": "Budi",
    "desk_c": "Siti"
  },
  "zones": {
    "angle_1": {
      "desk_a": { "x": 20, "y": 65, "w": 85, "h": 110 },
      "desk_b": { "x": 115, "y": 65, "w": 85, "h": 110 },
      "desk_c": { "x": 210, "y": 65, "w": 85, "h": 110 }
    }
  },
  "classes": ["empty", "occupied"],
  "input_size": [96, 96]
}
```

### Model Configuration

Edit constants in the Python files:

- `IMG_SIZE`: Input image size (default: 96x96)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Training epochs (default: 20)
- `LEARNING_RATE`: Optimizer learning rate (default: 0.001)

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- opencv-python
- Pillow
- tqdm
- scikit-learn
- numpy

## Hardware Requirements

- GPU recommended for training (CUDA)
- CPU inference is fast enough for real-time use

## Troubleshooting

### "No images found" when running auto_crop.py

- Check if images have uppercase extensions (.JPG, .PNG)
- The code handles case-insensitive extensions automatically

### "No zone definitions for angle_1"

- Run `python quick_label.py --angle angle_1` first to define zones
- Check that zones.json exists and has correct format

### Low accuracy

- Ensure manual labeling is consistent
- More training images generally improve accuracy
- Try adjusting the confidence threshold with `--threshold 0.6`

## License

Student project - use freely for educational purposes.
