# Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices

Official implementation of the ICCV/EVW 2025 paper  
**"Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices"**  
by *Mehdi Abbassi, Alberto Ancilotto, and Elisabetta Farella* —  
Energy Efficient Embedded Digital Architectures (E3DA) Unit, **Fondazione Bruno Kessler**.

---

## 🧠 Overview
This repository implements **Contextual Convolutions (ContextualConv blocks)** — a lightweight, biologically inspired mechanism for **forward-only learning**.  
Unlike conventional backpropagation-based training, Contextual Convolutions enable **block-wise, context-driven learning** using **context vectors** instead of gradients.  

Each block processes both the input feature maps and an accompanying **context signal**, allowing incremental, interpretable, and memory-efficient training suitable for **tiny and edge devices**.

---

## 📁 Repository Structure
```

├── main.py                 # Main training pipeline (Hydra entry point)
├── train_block.py          # Forward-only block training logic
├── contextual_conv.py      # Core ContextualConv2d implementation
├── blocks.py               # ContextualConvBlock, PoolConv, and classifier modules
├── data.py                 # Data loading and splitting utilities
├── data_gen.py             # Contextual data generation module
├── model_factory.py        # Model assembly from configuration
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

Run the **forward-only sequential training** pipeline:

```bash
python main.py
```

Each block — starting from the first **ContextualConv block** — is trained **independently** using **contextual supervision**, without backpropagation across the network.

### 3. Monitor Training

All training metrics (loss, block accuracy, data shapes) are logged to [Weights & Biases](https://wandb.ai).

---

## 🧩 Core Modules

### 🧱 `blocks.py` – ContextualConv Blocks

Defines the **ContextualConvBlock**, the fundamental unit of Contextual Convolutions.
Each block consists of:

* Optional **max pooling** (`pool_size > 1`)
* A **ContextualConv2d** layer that integrates both image features and contextual signals
* **Layer normalization** and **ReLU activation**

Example:

```python
from blocks import ContextualConvBlock
block = ContextualConvBlock(c_in=3, c_out=16, kernel_size=3, context_dim=10, pool_size=2)
output = block(images, context)
```

Also includes:

* `PoolConv` – a standard non-contextual convolutional block (used in later layers)
* `GlobalAvgPoolClassifier` – global average pooling followed by a linear classifier

---

### 🔁 `train_block.py` – Forward-Only Block Training

Implements **contextual forward-only training**:

* Each block learns to discriminate **contextually correct vs incorrect** image–context pairs.
* Uses **binary classification loss (`BCEWithLogitsLoss`)**.
* Trains blocks **sequentially** to emulate progressive, local learning.

Run standalone:

```bash
python train_block.py
```

---

### 🧩 `data_gen.py` – Contextual Data Generation

Generates **contextual positive and negative samples**:

* **Positive context:** image paired with its true label (one-hot encoded)
* **Negative context:** image paired with a random incorrect label
* Produces augmented batches `(2B, C, H, W)` with matching context tensors `(2B, num_classes)`

Example:

```python
from data_gen import generate_data, DataGenConfig
aug_images, aug_contexts = generate_data((images, labels), DataGenConfig())
```

---

## 🧪 Reproducibility

All experiments are fully configurable and reproducible via Hydra:

```bash
python main.py seed=42 optimizer=sgd lr=0.005 epochs=20
```

---

## 💡 Summary

| Component               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **ContextualConvBlock** | Learns from feature–context pairs via ContextualConv2d       |
| **train_block.py**      | Sequential forward-only training with contextual supervision |
| **data_gen.py**         | Context-based data augmentation (positive/negative contexts) |
| **Hydra + WandB**       | Reproducible configuration and experiment tracking           |

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
