# Setup (PyTorch, paths, device, seeds)
import os, random, math, time
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import torch
import torch.nn as nn
import shutil
import tempfile
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms ,models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from contextlib import nullcontext
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15

# ---------- utils ----------
TRAIN_VAl = ["train", "val"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(image_folder_dataset) -> float:
    # ImageFolder: class_to_idx alphabetical => NORMAL=0, PNEUMONIA=1
    labels = [y for _, y in image_folder_dataset.samples]
    neg = (np.array(labels) == 0).sum()
    pos = (np.array(labels) == 1).sum()
    pos_weight = torch.tensor([neg / max(pos,1.0)], dtype=torch.float32)
    print(f"Train counts -> NEG(NORMAL)={neg}, POS(PNEUMONIA)={pos}, pos_weight={pos_weight.item():.3f}")
    return pos_weight


def copy_files(file_paths, labels, dest_dir):
    for path, label in zip(file_paths, labels):
        dest = os.path.join(dest_dir, label)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(path, dest)


def show_split_counts(dir, title):
    print(f"{title}:")
    for split in TRAIN_VAl:
        for cls in CLASS_NAMES:
            folder = os.path.join(dir, split, cls)
            print(f"{split}/{cls}: {len(os.listdir(folder))}")
    plot_distribution_from_folder(dir, title)


def balance_val_train_split(dir):
    all_data = []
    for split in TRAIN_VAl:
        for cls in CLASS_NAMES:
            folder = os.path.join(dir, split, cls)
            for folder_name in os.listdir(folder):
                all_data.append((os.path.join(folder, folder_name), cls))

    # Stratified split: 85% train, 15% val
    paths, labels = zip(*all_data)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=42)

    # Use a writable temp directory for the split
    new_base = os.path.join(tempfile.gettempdir(), "chest_xray_split")
    if os.path.exists(new_base):
        shutil.rmtree(new_base)
    copy_files(train_paths, train_labels, os.path.join(new_base, 'train'))
    copy_files(val_paths, val_labels, os.path.join(new_base, 'val'))

    return new_base


# ---------- plots ----------
def plot_distribution_from_folder(base_dir, title_prefix):
    for split in TRAIN_VAl:
        counts = []
        for cls in CLASS_NAMES:
            folder = os.path.join(base_dir, split, cls)
            counts.append(len(os.listdir(folder)))
        plt.figure(figsize=(6,4))
        plt.bar(CLASS_NAMES, counts)
        plt.title(f"{title_prefix} {split} set class counts")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def plot_curves(h):
    fig, axs = plt.subplots(1,3, figsize=(16,4))
    axs[0].plot(h["train_loss"], label="train"); axs[0].plot(h["val_loss"], label="val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(h["train_acc"], label="train"); axs[1].plot(h["val_acc"], label="val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    axs[2].plot(h["train_auc"], label="train"); axs[2].plot(h["val_auc"], label="val")
    axs[2].set_title("ROC-AUC"); axs[2].legend()
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, lw=2);
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('FPR');
    plt.ylabel('TPR');
    plt.title('ROC Curve (Test)')
    plt.grid(True);
    plt.show()


# ---------- CNN model ----------
class PneumoCNN(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(PneumoCNN, self).__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(m.fc.in_features, 1)
        )
        self.net = m

    def forward(self, x):
        return self.net(x).squeeze(1)  # [B]


# ---------- train/eval core ----------
def _run_epoch(model, loader, criterion, *, is_train, optimizer=None, acc_threshold=0.5, epochNumber=0, tag=""):
    """Core loop used by both train and eval."""
    model.train() if is_train else model.eval()
    ctx = nullcontext() if is_train else torch.inference_mode()

    epoch_loss, correct, n = 0.0, 0, 0
    all_probs, all_targets = [], []

    with ctx:
        for images, targets in tqdm(loader, desc=f"[{tag}] Epoch {epochNumber}/{EPOCHS}"):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()

            logits = model(images).squeeze(1)
            loss   = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= acc_threshold).long()
            correct   += (preds == targets.long()).sum().item()
            n         += targets.size(0)
            epoch_loss += loss.item() * targets.size(0)

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = epoch_loss / n
    acc = correct / n
    probs_np = torch.cat(all_probs).numpy()
    targets_np = torch.cat(all_targets).numpy()

    try:
        auc = roc_auc_score(targets_np, probs_np)
    except Exception:
        auc = float("nan")

    return avg_loss, acc, auc, probs_np, targets_np


def train_one_epoch(model, loader, epochNumber, optimizer, criterion):
    return _run_epoch(model, loader, criterion,
                      is_train=True, optimizer=optimizer,
                      acc_threshold=0.5, epochNumber=epochNumber, tag="Train")


def eval_one_epoch(model, loader, epochNumber, criterion):
    return _run_epoch(model, loader, criterion,
                      is_train=False, optimizer=None,
                      acc_threshold=0.5, epochNumber=epochNumber, tag="Eval")


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=15, patience=10):
    best_val_auc, no_improve = -np.inf, 0
    ckpt_path = "best_cnn_pneumonia.pt"
    history = {"train_loss":[], "train_acc":[], "train_auc":[],
               "val_loss":[], "val_acc":[], "val_auc":[]}

    start = time.time()
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_auc, _, _ = train_one_epoch(model, train_loader, epoch, optimizer, criterion)
        val_loss, val_acc, val_auc, _, _ = eval_one_epoch(model, val_loader, epoch, criterion)

        if scheduler is not None:
            scheduler.step(val_auc)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc); history["train_auc"].append(tr_auc)
        history["val_loss"].append(val_loss);   history["val_acc"].append(val_acc);   history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss {tr_loss:.4f} Accuracy {tr_acc:.4f} AUC {tr_auc:.4f} | "
              f"Val Loss {val_loss:.4f} Accuracy {val_acc:.4f} AUC {val_auc:.4f}")

        # Early stopping & checkpoint
        if val_auc > best_val_auc:
            best_val_auc, no_improve = val_auc, 0
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val AUC={best_val_auc:.4f}")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Training finished in {(time.time()-start)/60:.1f} min. Best val AUC: {best_val_auc:.4f}")
    return history, best_val_auc


# Predict image function
def predict_image(path, model, tfms, threshold=0.5):
    model.eval()
    img = Image.open(path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)
    return prob, pred  # P(pneumonia), predicted label


# ---------- operating point & test ----------
def pick_threshold_from_val(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)  # thr len = len(prec)-1
    f1 = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    return float(thr[np.argmax(f1)])


def evaluate_with_threshold(model, loader, criterion, threshold):
    loss, acc05, auc, probs, targets = eval_one_epoch(model, loader, epochNumber=0, criterion=criterion)
    preds = (probs >= threshold).astype(int)
    acc_thr = (preds == targets).mean()
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-12)
    sens = tp / (tp + fn + 1e-12)
    bal_acc = (spec + sens) / 2
    pr_auc = average_precision_score(targets, probs)

    print(f"TEST — loss: {loss:.4f} | acc@thr: {acc_thr:.4f} | auc: {auc:.4f}")
    print(f"Specificity: {spec:.4f} | Sensitivity: {sens:.4f} | "f"Balanced Acc: {bal_acc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(targets, preds, target_names=CLASS_NAMES, digits=4))

    # ROC curve (test)
    fpr, tpr, _ = roc_curve(targets, probs)
    plot_roc_curve(fpr, tpr)

    # return {"loss":loss, "acc@thr":acc_thr, "auc":auc,
    #         "specificity":spec, "sensitivity":sens,
    #         "balanced_acc":bal_acc, "pr_auc":pr_auc,
    #         "cm":cm, "probs":probs, "targets":targets}