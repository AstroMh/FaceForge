# 🧪 DCGAN on CelebA (WIP)

> Work in progress — training a **Deep Convolutional GAN (DCGAN)** to generate 64×64 human face images using **TensorFlow/Keras**.  
> Built by **Mohammad Hemmat** — Computer Engineering student | AI, Robotics & Cybersecurity Enthusiast.

---

## ✨ Project Goals
- Implement a reproducible **DCGAN** baseline on the **CelebA** dataset  
- Visualize generated faces during training  
- Save and restore model checkpoints for later continuation  
- Later steps: Add FID metric, better architectures, and stability improvements

---

## 🖇️ Status
- ✅ Data pipeline + preprocessing  
- ✅ Generator & Discriminator models  
- ✅ Checkpoints and sample saving  
- 🟨 Documentation and fine-tuning  
- ⏭️ Next: Implement evaluation metrics, add configuration system, and enhance generator quality

---

## 📦 Environment Setup

### Requirements
- Python 3.10+
- TensorFlow 2.12+ (GPU recommended)
- NumPy
- Matplotlib
- Pillow
- ImageIO

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/dcgan-celeba.git
cd dcgan-celeba

# Create a virtual environment
python -m venv .venv
# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


🧠 Model Overview
🧩 Generator

The generator transforms 100-dimensional random noise vectors into 64×64 RGB images using a sequence of transposed convolution layers.

Architecture Summary

Layer	Output Shape	Notes
Dense	(8×8×256)	Initial projection
Reshape	(8, 8, 256)	Start of upsampling
Conv2DTranspose (128)	(8, 8, 128)	Upsampling
Conv2DTranspose (64)	(16, 16, 64)	Upsampling
Conv2DTranspose (32)	(32, 32, 32)	Upsampling
Conv2DTranspose (3, tanh)	(64, 64, 3)	Output image
🧩 Discriminator

The discriminator distinguishes between real and generated images using convolutional layers and LeakyReLU activations.

Architecture Summary

Layer	Output Shape	Notes
Conv2D (64)	(32, 32, 64)	Feature extraction
Conv2D (128)	(16, 16, 128)	Deep features
Conv2D (256)	(8, 8, 256)	Compact representation
Flatten + Dense(1)	-	Real/fake prediction
⚙️ Training

Loss: Binary cross-entropy (from logits)

Optimizers: Adam (learning rate = 1e-4)

Batch Size: 128

Epochs: 50 (default)

Noise Dimension: 100


▶️ Quick Start

Run the training script:

python train.py


During training:

Checkpoints are stored in ./training_checkpoints/

Generated image grids are saved in ./samples/

A final sample grid is saved as final_image.png

Example console output:

Epoch 12, Generator Loss: 0.97, Discriminator Loss: 1.12, Time: 42.6 sec


👨‍💻 Author

Mohammad Hemmat
Scholarship Student at Peter the Great St. Petersburg Polytechnic University, Russia
📬 Telegram
 | Channel
 | LinkedIn
