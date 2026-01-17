# 🦖 AI Dinosaur Finder — Computer Vision Project

## 📌 Project Purpose

This project aims to build a **computer vision system** capable of identifying dinosaur species from images using a **Convolutional Neural Network (CNN)**.

The primary goal is **not raw performance**, but a **deep understanding of the full machine learning pipeline**:
from raw data acquisition to model inference, including architectural and technological decisions.

This project is intentionally designed as a **learning-by-building system**, not a production-ready AI.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (compatible with Python 3.13)
- pip for package management

### Setup Environment

1. **Verify Python version**:

   ```bash
   python --version  # Should show Python 3.11+ or 3.13+
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:

   ```bash
   # Linux/Mac:
   source venv/bin/activate

   # Windows:
   venv\Scripts\activate
   ```

4. **Install the package in editable mode** (includes all dependencies):

   ```bash
   pip install -e .
   ```

   This will install:
   - PyTorch 2.9.1+ (CPU version)
   - torchvision 0.24.1+
   - Pillow 11.0.0
   - numpy, matplotlib, tqdm

5. **Verify installation**:

   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   python -c "import dino_finder; print('Dino Finder installed!')"
   ```

### Target Species

This project focuses on **3 visually distinct dinosaur species**:

- **Tyrannosaurus Rex** (T-Rex) - Iconic bipedal carnivore
- **Triceratops** - Herbivore with distinctive three-horned skull
- **Pteranodon** - Flying reptile (for morphological diversity)

These species were chosen for their:

- Visual distinctiveness (easier for CNN to learn differences)
- Popularity (more images available)
- Morphological diversity (bipedal, quadrupedal, flying)

### Getting the Dataset

#### Option 1: Manual Download (Recommended for Learning)

1. Create the directory structure if it doesn't exist:
   ```bash
   mkdir -p data/raw/{trex,triceratops,pteranodon}
   ```

2. Download images manually:
   - Search Google Images for each species
   - Download 10-50 images per species&
   - Save them in the corresponding folders:
     - `data/raw/trex/` → trex_001.jpg, trex_002.jpg, ...
     - `data/raw/triceratops/` → triceratops_001.jpg, ...
     - `data/raw/pteranodon/` → pteranodon_001.jpg, ...

3. Verify your dataset:
   ```bash
   python test.py
   ```

#### Option 2: Automated Download (Experimental)

The project includes a download script, but it may be blocked by search engines:

```bash
python scripts/download_images.py --num-images 50
```

**Note**: Web scraping can be unreliable. Manual download ensures better quality control.

---

## 🧱 High-Level System Overview

Raw Images
↓
Dataset Cleaning & Validation
↓
Preprocessing & Data Augmentation
↓
CNN Architecture
↓
Training & Optimization
↓
Evaluation
↓
Inference (Prediction + Confidence)


Each stage is implemented explicitly to maintain visibility over the entire system.

---

## 🛠️ Technology Choices

### Language: Python 3.11.14

Python is used as a **control and orchestration language**, not for performance-critical computation.

**Version choice**: Python 3.11.14

- Stable and mature (not bleeding-edge like 3.13)
- Officially supported by PyTorch ecosystem
- Performance improvements over 3.10 while maintaining stability
- Balance between modern features and compatibility

Key reasons for Python:

- The deep learning ecosystem is mature and stable
- Enables rapid experimentation and debugging
- Allows focus on ML system behavior rather than low-level engine implementation

**Important note**:
All performance-critical operations (tensor computation, GPU execution, backpropagation) are executed in **C/C++ and CUDA under the hood**.
Python acts as an expressive interface to these optimized components.

---

### Deep Learning Framework: PyTorch 2.9.1+

PyTorch is chosen for its **dynamic execution model** and transparency.

**Version choice**: PyTorch 2.9.1+ (CPU by default, GPU compatible)

- Latest stable release with extensive documentation and community support
- CPU-only for learning phase (can migrate to GPU later without code changes)
- Mature and production-ready with modern features
- CUDA libraries included but optional for GPU acceleration

Reasons for PyTorch:

- **Imperative programming style**: The code *is* the computation graph
- **Easier debugging and introspection**: Can inspect tensors and gradients at any point
- **Strong alignment with system-level reasoning**: Less abstraction hiding
- **Widely adopted in research and experimentation**: Better for understanding internals

**Why not TensorFlow/Keras?**
TensorFlow/Keras were intentionally avoided to reduce abstraction opacity during learning. PyTorch's explicit approach better serves the goal of understanding how CNNs work.

---

### Supporting Libraries

- `torchvision`: datasets, transforms, vision utilities
- `Pillow`: image loading and manipulation
- `numpy`: numerical operations
- `matplotlib`: visualization and debugging
- `argparse`: clean CLI for inference scripts

---

## 🧬 Architectural Choices

### Model Architecture: Custom CNN

A **custom Convolutional Neural Network** is implemented from scratch instead of using a pre-trained model.

#### Rationale:
- To understand how visual features are progressively extracted
- To observe how architectural depth and width affect learning
- To avoid treating the model as a black box

### High-Level Architecture

Input Image
↓
Convolution + ReLU + Batch Normalization
↓
Max Pooling
↓
Convolution + ReLU + Batch Normalization
↓
Max Pooling
↓
Convolution + ReLU
↓
Flatten
↓
Fully Connected Layer(s)
↓
Softmax Output


### Design Principles:
- Gradual feature abstraction
- Controlled depth to avoid overfitting
- Explicit separation of concerns

---

## 📂 Project Structure

```text
AI-Dino-Finder/
├── src/                       # Source code (importable package)
│   └── dino_finder/          # Main package
│       ├── models/           # CNN architecture definitions
│       ├── data/             # Dataset and data loading
│       ├── training/         # Training loop and logic
│       └── utils/            # Utilities (visualization, config)
├── scripts/                   # Executable scripts (entry points)
│   ├── download_images.py    # Data collection
│   ├── preprocess.py         # Preprocessing pipeline
│   ├── train.py              # Model training
│   └── predict.py            # Inference on new images
├── data/                      # Datasets (gitignored)
│   ├── raw/                  # Original images by species
│   │   ├── trex/
│   │   ├── triceratops/
│   │   └── pteranodon/
│   ├── train/                # Training set (70%)
│   ├── val/                  # Validation set (15%)
│   └── test/                 # Test set (15%)
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks (exploration)
├── checkpoints/               # Model checkpoints (gitignored)
├── README.md
├── requirements.txt
└── setup.py                   # Package installation config
```

**Design Philosophy**: Professional Python package structure

- `src/` contains reusable library code
- `scripts/` contains executable entry points
- Clear separation between code and data
- Follows Python packaging best practices

---

## 🧪 Dataset Strategy

- Images are collected via web scraping
- Each species has approximately the same number of samples
- Dataset is manually cleaned to remove irrelevant or low-quality images
- No external benchmark dataset is used

This ensures full control over data quality and biases.

---

## 📊 Training & Evaluation

### Loss Function
- Cross-Entropy Loss (multi-class classification)

### Optimizer
- Adam (initial choice for stability)

### Metrics
- Training and validation accuracy
- Training and validation loss
- Qualitative evaluation via visual inspection

---

## 🔍 Inference

A standalone inference script allows:
- Passing a single image as input
- Receiving a predicted species
- Displaying a confidence score

This ensures the model is usable independently of the training pipeline.

---

## ⚠️ Known Limitations

- Small dataset size
- Potential dataset bias (image sources)
- No guarantee of real-world generalization
- Simplified taxonomy of dinosaur species

These limitations are **acknowledged and intentional** within the scope of the project.

---

## 🚀 Possible Extensions

- Transfer learning comparison (ResNet, etc.)
- Confusion matrix visualization
- Feature map visualization
- Deeper architectural experimentation
- Dataset expansion

---

## 📖 Learning Outcomes

This project focuses on developing:
- System-level understanding of computer vision
- Practical knowledge of CNN internals
- Ability to justify architectural and technological choices
- Critical analysis of model behavior and limitations

---

## 🧠 Final Note

This project is not designed to impress with performance,
but to demonstrate **clarity of thought, technical rigor, and system understanding**.

Every decision is deliberate.
Every abstraction is questioned.
Every limitation is acknowledged.
