import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from contextlib import nullcontext
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from xray_helpers import plot_roc_curve, CLASS_NAMES


# ---------- CNN model ----------
class PneumoCNN(nn.Module):
    def __init__(self, dropout_p=0.4):
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
def _run_epoch(model, loader, criterion, device, is_train, optimizer=None,
               acc_threshold=0.5, epochNumber=0, epochs=None, tag=""):
    """Core loop used by both train and eval."""
    model.train() if is_train else model.eval()
    ctx = nullcontext() if is_train else torch.inference_mode()

    epoch_loss, correct, n = 0.0, 0, 0
    all_probs, all_targets = [], []

    with ctx:
        for images, targets in tqdm(loader, desc=f"[{tag}] Epoch {epochNumber}/{epochs}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= acc_threshold).long()
            correct += (preds == targets.long()).sum().item()
            n += targets.size(0)
            epoch_loss += loss.item() * targets.size(0)

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = epoch_loss / n
    acc = correct / n
    probs_np = torch.cat(all_probs).numpy()
    targets_np = torch.cat(all_targets).numpy()

    # ROC-AUC
    try: auc = roc_auc_score(targets_np, probs_np)
    except Exception: auc = float("nan")

    # PR-AUC (Average Precision)
    try: pr_auc = average_precision_score(targets_np, probs_np)
    except Exception: pr_auc = float("nan")

    return avg_loss, acc, auc, pr_auc, probs_np, targets_np


def train_one_epoch(model, loader, epochNumber, optimizer, criterion, device, epochs):
    return _run_epoch(model, loader, criterion, device,
                      is_train=True, optimizer=optimizer,
                      acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Train")


def eval_one_epoch(model, loader, epochNumber, criterion, device, epochs):
    return _run_epoch(model, loader, criterion, device,
                      is_train=False, optimizer=None,
                      acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Eval")


def fit(model, train_loader, val_loader, criterion, optimizer, device, scheduler=None, epochs=15, patience=10):
    best_val_auc, no_improve = -np.inf, 0
    ckpt_path = "best_cnn_pneumonia.pt"
    history = {"train_loss": [], "train_acc": [], "train_auc": [], "train_pr_auc": [],
               "val_loss": [], "val_acc": [], "val_auc": [], "val_pr_auc": []}

    start = time.time()
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_auc, tr_pr, _, _ = train_one_epoch(model, train_loader, epoch, optimizer, criterion, device, epochs)
        val_loss, val_acc, val_auc, val_pr, _, _ = eval_one_epoch(model, val_loader, epoch, criterion, device, epochs)

        if scheduler is not None:
            scheduler.step(val_auc)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc); history["train_pr_auc"].append(tr_pr); history["train_auc"].append(tr_auc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc); history["val_pr_auc"].append(val_pr); history["val_auc"].append(val_auc)

        print(f"Train Loss: {tr_loss:.4f}, Accuracy: {tr_acc:.4f}, AUC: {tr_auc:.4f}, PR {tr_pr:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, PR {val_pr:.4f}")

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

# ---------- operating point & test ----------
# def pick_threshold_from_val(y_true, y_prob):
#     prec, rec, thr = precision_recall_curve(y_true, y_prob)  # thr len = len(prec)-1
#     f1 = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
#     return float(thr[np.argmax(f1)])


def pick_threshold_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = j[1:].argmax() + 1  # thresholds align with fpr/tpr[1:]
    return float(thr[idx-1])


def evaluate_with_threshold(model, loader, criterion, threshold, device, epochs):
    loss, acc05, auc, probs, targets = eval_one_epoch(model, loader, 1, criterion, device, epochs)
    # Print first 10 probabilities and targets for inspection
    print("probs[:10] =", probs[:10])
    print("targets[:10] =", targets[:10])

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