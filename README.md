# Skin-AI-Lab: Skin Cancer Detection & Data Augmentation using DCGAN

Skin-AI-Lab is a research-oriented project that utilizes Deep Convolutional Generative Adversarial Networks (DCGAN) to generate synthetic skin lesion images. The project aims to improve the performance of skin cancer classifiers by augmenting real datasets with high-quality synthetic data, specifically addressing the challenge of data scarcity in medical imaging.

## 🚀 Overview

The project consists of three main phases:
1.  **Synthetic Data Generation**: Training a DCGAN on the HAM10000 dataset to create realistic synthetic skin lesion images.
2.  **Performance Evaluation**: Training a CNN classifier on real data vs. real data augmented with synthetic images to measure accuracy improvements.
3.  **Interactive Web Dashboard**: A Flask-based web interface to monitor training, trigger new training sessions, and generate images in real-time.

## 🛠️ Architecture

-   **Deep Convolutional GAN (DCGAN)**:
    -   **Generator**: Transpose convolutions to upsample noise into $64 \times 64$ RGB images.
    -   **Discriminator**: Convolutional layers with LeakyReLU and Dropout to distinguish real vs. fake images.
-   **Classifier**: A standard Convolutional Neural Network (CNN) for binary classification (Malignant vs. Benign).
-   **Web Interface**: Flask backend with a dynamic dashboard for process management.

## 📂 Project Structure

```text
├── app.py                      # Flask web application
├── dcgan_train.py              # Core training script for GAN and Classifier
├── HAM10000_metadata.csv       # Dataset metadata
├── setup.sh                    # WSL/Linux environment setup script
├── run_training.bat            # Windows entry point for training
├── run_tf.sh                   # TensorFlow execution script
├── test_gpu.py / test_gpu.sh   # Utility scripts for GPU verification
├── templates/
│   └── index.html              # Dashboard frontend
├── data/                       # (Ignored) Source dataset images
└── synthetic_images/           # (Generated) Synthetic output images
```

## ⚙️ Installation & Setup

### Prerequisites
-   Windows with **WSL2** (Ubuntu-22.04 recommended).
-   NVIDIA GPU with CUDA support for accelerated training.
-   Python 3.10+

### Step-by-Step Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/esatyatej1/Skin-AI-Lab.git
    cd Skin-AI-Lab
    ```

2.  **Dataset Preparation**:
    -   Download the HAM10000 dataset.
    -   Place images in `c:/123/data/images`.
    -   Ensure `HAM10000_metadata.csv` is in the root directory.

3.  **Environment Setup**:
    The project uses a dedicated virtual environment within WSL. You can initialize it using the provided batch script:
    -   Run `run_training.bat` in Windows.
    -   This script triggers `setup.sh` inside WSL, installs dependencies (TensorFlow, OpenCV, Pandas), and starts the training.

## 🖥️ Usage

### Web Dashboard
Start the Flask server to access the GUI:
```bash
python app.py
```
Open `http://localhost:5000` to:
-   **Monitor Logs**: Real-time training progress.
-   **Generate Images**: Trigger the GAN to create a new skin lesion image.
-   **Control Training**: Start/Stop the training process remotely.

### Training Manually
If you prefer running the training script directly:
```bash
# Inside WSL
source /mnt/c/123/train_env/bin/activate
python3 /mnt/c/123/dcgan_train.py
```

## 📊 Results & Evaluation

The project evaluates the model by comparing validation accuracy:
-   **Base Accuracy**: CNN trained only on real images.
-   **Augmented Accuracy**: CNN trained on a mix of real and DCGAN-generated images.

Preliminary results show that adding synthetic data helps the model generalize better, especially for underrepresented classes in the dataset.

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
**Disclaimer**: This project is for educational and research purposes only. It is not intended for clinical diagnosis.
