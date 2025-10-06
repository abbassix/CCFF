# Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices

Official implementation of the ICCV/EV 2025 paper **"Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices"** by *Mehdi Abbassi, Alberto Ancilotto, and Elisabetta Farella* — Energy Efficient Embedded Digital Architectures (E3DA) unit at the Fondazione Bruno Kessler.

---

## 🧠 Overview
This repository contains the implementation of **Contextual Convolutions (CCs)**, a lightweight, biologically inspired alternative to backpropagation designed for **forward-only learning**.  
CCs enable **incremental block-wise training** with **contextual signals** that guide feature learning, making them suitable for **tiny and resource-constrained devices**.

Our training framework allows models to be trained **sequentially, one block at a time**, without the need for end-to-end backpropagation or gradient storage.

---

## 📁 Repository Structure
```

├── main.py                 # Main training pipeline (Hydra entry point)
├── train_block.py          # Block-wise training logic
├── blocks.py               # PoolConv module definitions
├── data.py                 # Data loading and splitting utilities
├── data_gen.py             # Contextual data generation module
├── model_factory.py        # Model assembly from config
└── config.yaml             # Hydra configuration file

````

---

## ⚙️ Installation
### Requirements
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- torchvision  
- hydra-core  
- wandb  
- omegaconf  

Install all dependencies:
```bash
pip install torch torchvision hydra-core wandb omegaconf
````

---

## 🚀 Usage

### 1. Configure Training

Modify parameters in `config.yaml`:

```yaml
dataset: "cifar100"
data_gen_type: "contextual"
epochs: 10
optimizer: "adam"
lr: 0.001
```

### 2. Train the Model

Run the sequential forward-only training pipeline:

```bash
python main.py
```

Each block of the model will be trained **independently and sequentially** using **contextual data**.

### 3. Monitor Training

Training metrics (loss, block performance, data shapes) are logged automatically to [Weights & Biases](https://wandb.ai).

---

## 🧩 Core Modules

### 🧱 `blocks.py` – PoolConv Blocks

Defines **PoolConv**, a simple convolutional block consisting of:

* Optional **max pooling** with kernel size `m×m`.
* A **convolution** layer with `c_out` filters of size `k×k`.

Example:

```python
from blocks import PoolConv
block = PoolConv(c_in=3, m=2, c_out=16, k=3)
```

---

### 🔁 `train_block.py` – Forward-Only Block Training

Implements **sequential block training** with binary classification loss:

* Each block is trained to distinguish between **contextually correct** and **incorrect** pairs.
* Uses `nn.BCEWithLogitsLoss` and logs progress via wandb.

Run standalone:

```bash
python train_block.py
```

---

### 🧩 `data_gen.py` – Contextual Data Generation

Generates **contextual positive and negative samples** using label-based augmentation:

* **Positive context:** image paired with its true class.
* **Negative context:** same image paired with a random wrong class.

If the input batch is `(B, C, H, W)`, the output becomes `(2B, C, H, W)` with a context tensor `(2B, num_classes)`.

Example:

```python
from data_gen import generate_data, DataGenConfig
aug_images, aug_contexts = generate_data((images, labels), DataGenConfig())
```

---

## 🧪 Reproducibility

All experiments are controlled via Hydra configurations for reproducibility:

```bash
python main.py seed=42 optimizer=sgd lr=0.005 epochs=20
```

<!-- ---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{abbassi2025contextual,
  title={Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices},
  author={Abbassi, Mehdi and Ancilotto, Alberto and Farella, Elisabetta},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
``` -->
