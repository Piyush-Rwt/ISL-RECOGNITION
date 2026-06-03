# Project ATHENA: Indian Sign Language (ISL) Recognition

Project **ATHENA** is a high-performance, real-time Indian Sign Language (ISL) recognition system. It leverages computer vision and deep learning to translate hand gestures into text (0-9, A-Z) with high stability and accuracy.

---

## 🚀 Key Features

*   **Real-Time Performance**: Processes 30+ FPS on a standard laptop CPU using a lightweight MLP architecture.
*   **Geometric Feature Engineering**: Uses a custom 180-dimensional feature vector instead of raw pixels, making it robust against background noise and lighting.
*   **Invariance**: Robust to hand position (**Translation Invariance**) and hand-to-camera distance (**Scale Invariance**).
*   **Advanced Inference**: Employs **Test-Time Augmentation (TTA)** and **Weighted Voting** to eliminate flickering and provide stable predictions.
*   **Multi-Hand Support**: Capable of processing both right and left hand inputs simultaneously.

---

## 🛠️ Technical Stack

*   **Language**: Python 3.x
*   **Frameworks**: PyTorch (Deep Learning), MediaPipe (Hand Tracking), OpenCV (Computer Vision).
*   **Libraries**: NumPy, Scikit-learn, Matplotlib, Tqdm.

---

## 🧠 System Architecture

### 1. Data Pipeline
The system follows a modular pipeline:
`Camera Input` → `MediaPipe Hands` → `Normalization` → `Feature Engineering` → `MLP Inference` → `Weighted Voting` → `Final Output`.

### 2. Feature Engineering (180-Dim Vector)
Instead of feeding images directly, we extract 90 features per hand (Total 180):
*   **Normalized XYZ (63):** 21 landmarks centered at the wrist and scaled relative to the hand size.
*   **Derived Geometric Features (27):**
    *   **Bone Angles (10):** Computing the angular relationship between finger segments using `arccos` of dot products.
    *   **Fingertip Distances (10):** Pairwise Euclidean distances between all 5 fingertips.
    *   **Thumb-to-Finger Distances (4):** Distance from the thumb tip to all other finger tips.
    *   **Palm Normal (3):** A 3D vector representing the palm's orientation using the cross product of the hand plane.

### 3. Model Architecture (v8)
The core is a Deep Multi-Layer Perceptron (MLP) with 5 hidden layers:
*   **Input**: 180 features
*   **Hidden Layers**: 512 → 512 → 256 → 128 → 64
*   **Regularization**: **Batch Normalization** after every layer and **Dropout** (0.4 to 0.2) to prevent overfitting.
*   **Output**: 36 classes (Softmax).

---

## 📈 Training Techniques

*   **Data Augmentation**: Includes XY rotation ($\pm 20°$), random scaling (0.8x-1.2x), wrist jitter, mirror flipping, and **Finger Dropout** (randomly "hiding" a finger to force the model to learn from partial data).
*   **Class Weighting**: Balanced `CrossEntropyLoss` to handle class imbalances in the ISLRTC and Prekshapalva datasets.
*   **Optimization**: **Adam Optimizer** with **Weight Decay (L2 Regularization)**.
*   **Learning Rate Control**: `ReduceLROnPlateau` scheduler to fine-tune weights as the model nears convergence.

---

## 🖥️ Live Inference Logic

*   **Test-Time Augmentation (TTA)**: During inference, the system runs 7 parallel versions of the input with slight noise and averages the results for higher precision.
*   **Weighted Voter**: A temporal buffer (size 20) stores recent predictions. Newer frames and high-confidence results are given higher weights to produce a "locked" and jitter-free display.
*   **Confidence Thresholding**: Predictions are only displayed if the model is $>45\%$ confident, preventing "random guessing" when no hand is present.

---

## 📂 Project Structure

*   `train_athena_v8.py`: The complete training script with data loading, augmentation, and model definition.
*   `test_webcam_v8.py`: Real-time inference script for webcam usage.
*   `alphabet_mlp_v8.pth`: The pre-trained PyTorch model weights.
*   `data/`: Directory for dataset and feature caches.

---

## 🛠️ How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch mediapipe opencv-python numpy sklearn tqdm
    ```
2.  **Run Inference**:
    ```bash
    python test_webcam_v8.py
    ```
3.  **To Train**:
    Update the `BASE_DIR` in `train_athena_v8.py` and run:
    ```bash
    python train_athena_v8.py
    ```

---

## 🔮 Future Scope
*   **Temporal Integration**: Adding LSTM or GRU layers to recognize dynamic signs and full sentences.
*   **Bilingual Support**: Adding support for American Sign Language (ASL) alongside ISL.
*   **Mobile Deployment**: Converting the model to ONNX for Android/iOS integration.

---
**Developed by [Piyush Rawat]** | *Project ATHENA - Empowering communication through technology.*
