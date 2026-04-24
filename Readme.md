# 🎯 Real-Time Facial Recognition System

A high-performance real-time facial recognition system using LBPH with adaptive thresholding and open-set unknown face rejection.

---

## 🚀 Key Features

* Real-time face detection and recognition using OpenCV
* ~96% accuracy on known faces
* ~85% unknown face rejection rate
* Adaptive multi-threshold decision system
* Temporal smoothing for stable predictions
* Data normalization and duplicate handling

---

## 💡 Core Innovation

This project solves a key limitation in traditional LBPH systems.

Instead of treating **"unknown" as a class**, this system uses an **open-set recognition approach**:

* Unknown faces are **rejected**, not classified
* Adaptive thresholding improves real-world reliability
* Reduces false positives significantly

---

## 🛠 Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-learn
* LBPH Algorithm

---

## ⚡ Quick Start

### ⚙️ Requirements

* Python 3.7+
* Webcam

---

## 📦 Dependencies

All required dependencies are listed in `requirements.txt`.

Main libraries:
- opencv-python
- numpy
- scikit-learn
- PyYAML
- Pillow

---

### 🧪 Setup

```bash
# Clone the repository
git clone https://github.com/CSE-ET/CSE-AI-ML-_Batch-20.git
cd major_project_final_version

# Create virtual environment
python -m venv venv

# Activate environment

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create required directories (if not present)
mkdir data models logs
```


---

### 📁 Required File

Ensure the Haar Cascade file is present in the `models/` directory.

If missing, download using:

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O models/haarcascade_frontalface_default.xml
```

---

### ▶️ Run

```bash
# Step 1: Capture images for a new user
python src/main.py --mode capture

# Step 2: Train the model
python src/main.py --mode train

# Step 3: Run real-time recognition
python src/main.py --mode recognize
```

⚠️ Note: You must capture and train the model before running recognition for the first time.

---

## 🎥 Demo

This system performs real-time facial recognition using a webcam, displaying:

* Detected faces with bounding boxes
* Predicted identity with confidence scores
* Automatic rejection of unknown faces in real-time
*(Demo media available upon request.)*

---

## 📊 Results

| Metric                 | Value |
| ---------------------- | ----- |
| Accuracy (Known Faces) | 96%   |
| Precision              | 95.8% |
| Recall                 | 95.1% |
| F1-score               | 95.4% |
| Unknown Rejection Rate | 85%+  |

---

## 📁 Project Structure

```
major_project_final_version/
│
├── data/                          # Dataset storage
│   ├── train/                     # Training images (organized by subject)
│   ├── test/                      # Test dataset for evaluation
│   └── unknown/                   # Unknown faces for rejection testing
│
├── models/                        # Model files
│   ├── lbph_model.yaml            # Trained LBPH model
│   ├── subjects_db.pkl            # Label mapping (name ↔ ID)
│   └── haarcascade_frontalface_default.xml  # Face detection model
│
├── src/                           # Core application code
│   ├── main.py                    # Entry point (CLI modes)
│   ├── detection.py               # Face detection logic
│   ├── preprocessing.py           # Image preprocessing (resize, CLAHE)
│   ├── training.py                # Model training pipeline
│   ├── recognition.py             # Recognition + threshold logic
│   ├── capture.py                 # Data collection module
│   └── gui.py                     # Real-time display interface
│
├── utils/                         # Utility modules
│   ├── config_loader.py           # Configuration handler
│   └── logger.py                  # Logging system
│
├── config.yaml                    # System configuration file
├── retrain.py                     # Data cleanup + retraining script
├── test_recognition.py            # Evaluation/testing script
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

### 📌 Structure Overview

* **data/** → Stores all training and testing images
* **models/** → Contains trained model and Haar Cascade file
* **src/** → Core logic of detection, training, and recognition
* **utils/** → Helper utilities (config + logging)
* **config.yaml** → Central configuration for thresholds and parameters
* **retrain.py** → Handles dataset cleanup and retraining
* **test_recognition.py** → Used for evaluating model performance

---

## 🧠 How It Works

1. Face Detection (Haar Cascade)
2. Preprocessing (Resize + CLAHE)
3. Feature Extraction (LBPH)
4. Prediction + Confidence Score
5. Threshold-Based Decision
6. Temporal Smoothing

---

## 🌍 Applications

* Smart attendance systems
* Secure access control
* Surveillance & monitoring
* Edge AI devices (Raspberry Pi)

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Performance drops with pose variations
* Requires sufficient training data

---

## 🔮 Future Improvements

* Integrate deep learning models (FaceNet, ArcFace)
* Add liveness detection
* Deploy as web/mobile application

---

## 📄 Documentation

Detailed project documentation is available upon request.

---

## 👩‍💻 Author

**Hasini Muvva**
B.Tech CSE (Final Year Project)

📧 [muvvahasiniraghu313@gmail.com](mailto:muvvahasiniraghu313@gmail.com)
🔗 https://github.com/CSE-ET/CSE-AI-ML-_Batch-20

---

## 📜 License

MIT License
