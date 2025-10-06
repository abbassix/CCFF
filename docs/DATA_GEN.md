# Data Generation Module (`data_gen.py`)

## 🧠 Overview
This module implements **contextual data generation** for the *Contextual Convolutions* framework, as described in our ICCV 2025 paper  
**"Contextual Convolutions for Scalable Forward-Only Learning on Tiny Devices"**.  

This method generates **context-aware pairs** of images and label contexts, enabling **forward-only block learning** without backpropagation.

---

## 🔍 Concept
The **contextual data generation** mechanism creates **positive** and **negative** pairs by associating each image with both its **correct** and a **wrong** class label.  
Each pair consists of:
- **Image** — the same visual input duplicated for positive and negative contexts.
- **Context vector** — a one-hot encoded class label representing either the correct or incorrect class.

This process enables the model to learn discriminative representations based purely on *contextual correctness*, aligning with the forward-only paradigm.

---

## ⚙️ Data Generation Process

### 1. Positive Contexts
Each image is paired with its **true label**, one-hot encoded:
```python
positive_context = one_hot_encode(labels, num_classes)
```

### 2. Negative Contexts

Each image is paired with a **wrong label**, randomly sampled from all other classes:

```python
negative_labels = generate_wrong_labels(labels, num_classes)
negative_context = one_hot_encode(negative_labels, num_classes)
```

### 3. Augmentation

Images are duplicated and concatenated:

```python
augmented_images = torch.cat([images, images], dim=0)
augmented_contexts = torch.cat([positive_context, negative_context], dim=0)
```

Final output shapes:

* **Images:** `(2B, C, H, W)`
* **Contexts:** `(2B, num_classes)`

---

## 🧩 Key Functions

### `generate_wrong_labels(true_labels, num_classes)`

Generates incorrect class labels for each sample.
Ensures that the wrong label is different from the true one.

**Args**

* `true_labels (Tensor)`: True labels `(B,)`
* `num_classes (int)`: Total number of classes

**Returns**

* `Tensor`: Wrong labels `(B,)`

---

### `one_hot_encode(labels, num_classes)`

Converts integer labels into one-hot encoded context vectors.

**Returns**

* `Tensor`: One-hot encoded contexts `(B, num_classes)`

---

### `generate_contextual_data(images, labels, num_classes)`

Builds positive and negative context-image pairs.

**Returns**

* `Tuple[Tensor, Tensor]`: `(augmented_images, augmented_contexts)`

---

### `generate_data(batch, cfg)`

Main entry point for contextual data generation.
Chooses the generation method based on the Hydra configuration field `cfg.data_gen_type`.

**Supported types**

* `"contextual"` — generates context-based samples (default)

Example:

```python
aug_images, aug_contexts = generate_data((images, labels), cfg)
```

---

## 🧰 Configuration

This module extends the base data configuration from `data.py` using the `DataGenConfig` dataclass:

```yaml
# Example config.yaml
dataset: "cifar100"
num_classes: 100
data_gen_type: "contextual"
batch_size: 64
```

You can switch between data generation modes (if others are added later) by modifying `data_gen_type`.

---

## 🚀 Usage Example

```python
from data_gen import generate_data, DataGenConfig
from data import load_and_split_data

cfg = DataGenConfig(dataset="cifar100", num_classes=100, data_gen_type="contextual")
train_loader, _, _ = load_and_split_data(cfg)

images, labels = next(iter(train_loader))
aug_images, aug_contexts = generate_data((images, labels), cfg)

print(aug_images.shape)     # (2B, C, H, W)
print(aug_contexts.shape)   # (2B, num_classes)
```

---

## 📊 Logging

The script integrates with **Weights & Biases (wandb)** for logging:

* Original batch shapes
* Augmented image/context shapes
* Configuration details

Automatically triggered when running:

```bash
python data_gen.py
```

---

## 💡 Summary

| Feature              | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| **Goal**             | Generate context-aware input-label pairs for forward-only learning |
| **Augmentation**     | Duplicate images + one-hot encode correct/wrong labels             |
| **Output**           | `(augmented_images, augmented_contexts)`                           |
| **Integration**      | Used inside the sequential block training loop (`train_block.py`)  |
| **Configurable via** | `config.yaml` and Hydra CLI overrides                              |
