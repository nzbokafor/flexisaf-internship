"""
FlexiSAF Module 4: Clothing Classifier (Fashion-MNIST)
Project chosen from DataCamp's "Computer Vision Projects" list (Beginner #6).

A small convolutional neural network (CNN) learns to recognise 10 types of
clothing from 28x28 grayscale images.

Steps:
  1. Load data   - download Fashion-MNIST (70,000 images) and scale pixels to 0-1
  2. Train       - fit a small CNN on the 60,000 training images
  3. Evaluate    - measure accuracy on the 10,000 unseen test images and save evidence
"""
import gzip, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EVIDENCE = ROOT / "evidence"
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
EPOCHS = 3
torch.manual_seed(42)


# ---------- Step 1: load data ----------
def load(name):
    path = DATA / FILES[name]
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print(f"Downloading {FILES[name]} ...")
        urllib.request.urlretrieve(URL + FILES[name], path)
    with gzip.open(path, "rb") as f:
        raw = np.frombuffer(f.read(), np.uint8)
    if name.endswith("_x"):
        return raw[16:].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    return raw[8:].astype(np.int64)


# ---------- Step 2: model and training ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    EVIDENCE.mkdir(exist_ok=True)
    x_train, y_train = load("train_x"), load("train_y")
    x_test, y_test = load("test_x"), load("test_y")
    log = [f"Training images: {len(x_train)}, test images: {len(x_test)}, classes: {len(CLASSES)}"]
    print(log[-1])

    train = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    model, loss_fn = CNN(), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        model.train(); total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train:
            opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); correct += (out.argmax(1) == yb).sum().item(); total += len(xb)
        log.append(f"Epoch {epoch}: loss {loss_sum/total:.3f}, training accuracy {100*correct/total:.1f}%")
        print(log[-1])

    # ---------- Step 3: evaluate ----------
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(x_test)).argmax(1).numpy()
    acc = 100 * (preds == y_test).mean()
    log.append(f"Test accuracy (unseen images): {acc:.1f}%")
    print(log[-1])
    log.append("Accuracy per class:")
    for i, c in enumerate(CLASSES):
        m = y_test == i
        log.append(f"  {c:<12} {100*(preds[m] == i).mean():.1f}%")
    print("\n".join(log[-11:]))

    # confusion matrix
    cm = np.zeros((10, 10), int)
    for t, p in zip(y_test, preds): cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix (test accuracy {acc:.1f}%)")
    fig.tight_layout(); fig.savefig(EVIDENCE / "confusion_matrix.png", dpi=120); plt.close(fig)

    # sample predictions
    fig, axes = plt.subplots(3, 6, figsize=(10, 5.5))
    for ax, i in zip(axes.flat, range(18)):
        ax.imshow(x_test[i, 0], cmap="gray"); ax.axis("off")
        ok = preds[i] == y_test[i]
        ax.set_title(f"{CLASSES[preds[i]]}\n(actual: {CLASSES[y_test[i]]})", fontsize=7, color="green" if ok else "red")
    fig.suptitle("Sample predictions on test images (green = correct, red = wrong)")
    fig.tight_layout(); fig.savefig(EVIDENCE / "sample_predictions.png", dpi=120); plt.close(fig)

    (EVIDENCE / "training_output.txt").write_text("\n".join(log) + "\n")
    torch.save(model.state_dict(), ROOT / "clothing_classifier" / "model.pt")


if __name__ == "__main__":
    main()
