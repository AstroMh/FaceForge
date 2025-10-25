# 🧪 DCGAN on CelebA (WIP)

> Work in progress — training a **DCGAN** to generate 64×64 face images using **TensorFlow/Keras**.  
> Built by **Mohammad Hemmat** (Computer Engineering student; AI, Robotics & Cybersecurity enthusiast).

---

## ✨ Project Goals
- Implement a clean DCGAN baseline on **CelebA**
- Reproducible training with checkpoints
- Save sample grids during training
- Later: improve stability, add metrics (FID), and experiment with architectures

---

## 🖇️ Status
- ✅ Data pipeline & DCGAN models (generator + discriminator)
- ✅ Checkpointing & sample image export
- 🟨 Docs & training tips
- ⏭️ Next: gradient penalty / spectral norm, FID evaluation, config file, CLI

---

## 📦 Environment

- Python 3.10+
- TensorFlow 2.12+ (GPU recommended)
- NumPy, Matplotlib, Pillow

```bash
# Create & activate a venv (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
