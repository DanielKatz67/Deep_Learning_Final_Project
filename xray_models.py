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

# ---------
# -----------------------------
# BatchNorm2d (custom)
# -----------------------------
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats)
    def forward(self, x):
        return self.bn(x)


# -----------------------------
# LayerNorm (2D input)
# -----------------------------
class LayerNorm2d(nn.Module):
    """
    Channel-first LayerNorm for (N, C, H, W): normalize over C for each (H,W) location uniformly.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=num_channels, eps=eps, elementwise_affine=True)

    def forward(self, x):
        # x: (N, C, H, W) -> move channels to last, apply LN over channels, then move back
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)            # (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        return x


# # -----------------------------
# # Vision Transformer (ViT)
# # -----------------------------
# class PatchEmbedding(nn.Module):
#     def _init_(self, in_channels, patch_size, embed_dim):
#         super()._init_()
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#     def forward(self, x):
#         x = self.proj(x)  # (B, E, H', W')
#         x = x.flatten(2)  # (B, E, H'*W')
#         x = x.transpose(1, 2)  # (B, N, E) where N is number of patches
#         return x
#
#
# class Attention(nn.Module):
#     def _init_(self, embed_dim, num_heads):
#         super()._init_()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
#         self.proj = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         return x
#
#
# class MLP(nn.Module):
#     def _init_(self, in_features, hidden_features, out_features):
#         super()._init_()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x
#
#
# class Block(nn.Module):
#     def _init_(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_rate=0.):
#         super()._init_()
#         self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.attn = Attention(embed_dim, num_heads)
#         self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
#         mlp_hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim)
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x
#
#
# class XRayViT(nn.Module):
#     def _init_(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12,
#                mlp_ratio=4., num_classes=1, dropout_p=0.4):
#         super()._init_()
#         self.num_patches = (img_size // patch_size) ** 2
#
#         self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
#
#         self.pos_drop = nn.Dropout(p=dropout_p)
#
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, mlp_ratio)
#             for _ in range(depth)
#         ])
#
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)
#
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#         return x[:, 0]
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x.squeeze(1)

# -----------------------------
# CNN model (BatchNorm or LayerNorm injected)
# -----------------------------
class XRayCNN(nn.Module):
    def __init__(self, num_classes, norm_layer):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,   32, norm_layer),
            self._block_mp(32, 32, norm_layer, kernel_size_mp=2),

            self._block(32,  64, norm_layer),
            self._block_mp(64, 64, norm_layer, kernel_size_mp=2),

            self._block(64, 128, norm_layer),
            self._block_mp(128, 128, norm_layer, kernel_size_mp=2),

            self._block(128, 256, norm_layer),
            # Removed last _block_mp to prevent feature map from becoming too small
            # self._block_mp(256, 256, norm_layer, kernel_size_mp=2),

            # nn.AdaptiveAvgPool2d((1,1))
        )

        # NEW: force to 1×1 spatially
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),          # (N,256,1,1) -> (N,256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64,  num_classes)
        )


    def _block(self, in_ch, out_ch, norm_layer, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )


    def _block_mp(self, in_ch, out_ch, norm_layer, kernel_size=3, stride=1, padding=1, kernel_size_mp=2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=kernel_size_mp)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ---------- ViT model ----------
def trunc_normal_(tensor, mean=0., std=0.02):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        # Conv2d does linear patch projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path1 = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.drop_path2 = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class XRayViT(nn.Module):
    """
    Vision Transformer for binary X-ray classification.
    Outputs ONE logit -> use BCEWithLogitsLoss(pos_weight=...).
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=8, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth decay across blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate,
                     attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 1-logit head for BCEWithLogitsLoss
        self.head = nn.Linear(embed_dim, 1)

        # init
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, E = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :N+1, :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out).squeeze(1)
        return logits


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
            targets = targets.to(device, non_blocking=True).float().view(-1)

            # if not hasattr(model, "_printed_input_shape"):
            #     print("Input batch shape:", images.shape)  # (B, C, H, W)
            #     model._printed_input_shape = True

            logits = model(images)

            # Ensure logits is [B] for BCEWithLogitsLoss
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            elif logits.ndim != 1:
                raise RuntimeError(f"Expected logits [B] or [B,1], got {tuple(logits.shape)}")


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


# def _run_epoch_with_sam(model, loader, criterion, device, is_train, optimizer=None,
#                acc_threshold=0.5, epochNumber=0, epochs=None, tag="",
#                w_neg: float = 1.0, w_pos: float = 1.0):
#     """Core loop used by both train and eval."""
#     model.train() if is_train else model.eval()
#     ctx = nullcontext() if is_train else torch.inference_mode()
#
#     epoch_loss, correct, n = 0.0, 0, 0
#     all_probs, all_targets = [], []
#
#     with ctx:
#         for images, targets in tqdm(loader, desc=f"[{tag}] Epoch {epochNumber}/{epochs}"):
#             images = images.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True).float().view(-1)
#
#             if not hasattr(model, "_printed_input_shape"):
#                 print("Input batch shape:", images.shape)  # (B, C, H, W)
#                 model._printed_input_shape = True
#
#             w_neg_t = torch.as_tensor(w_neg, device=device, dtype=targets.dtype)
#             w_pos_t = torch.as_tensor(w_pos, device=device, dtype=targets.dtype)
#
#             logits = model(images)
#             # Ensure logits is [B] for BCEWithLogitsLoss
#             if logits.ndim == 2 and logits.size(1) == 1:
#                 logits = logits.squeeze(1)
#             elif logits.ndim != 1:
#                 raise RuntimeError(f"Expected logits [B] or [B,1], got {tuple(logits.shape)}")
#
#             loss_vec = criterion(logits, targets)  # shape [B]
#             batch_w = torch.where(targets == 1, w_pos_t, w_neg_t)  # shape [B]
#             loss = (loss_vec * batch_w).mean()
#
#             # if is_train:
#             #     optimizer.zero_grad(set_to_none=True)
#             #     loss.backward()
#             #     optimizer.step()
#             if is_train:
#                 # ---- SAM: first step (ascent) ----
#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 optimizer.first_step(zero_grad=True)
#
#                 # ---- SAM: second step (descent) ----
#                 logits2 = model(images)
#                 if logits2.ndim == 2 and logits2.size(1) == 1:
#                     logits2 = logits2.squeeze(1)
#                 elif logits2.ndim != 1:
#                     raise RuntimeError(f"Expected logits [B] or [B,1], got {tuple(logits2.shape)}")
#
#                 loss2_vec = criterion(logits2, targets)  # [B]
#                 batch_w = torch.where(targets == 1, w_pos_t, w_neg_t)
#                 loss2 = (loss2_vec * batch_w).mean()
#
#                 loss2.backward()
#                 optimizer.second_step(zero_grad=True)
#
#                 logits = logits2.detach()  # use second forward for metrics
#                 loss_to_log = loss2
#             else:
#                 loss_to_log = loss
#
#             probs = torch.sigmoid(logits)
#             preds = (probs >= acc_threshold).long()
#             correct += (preds == targets.long()).sum().item()
#
#             bsz = targets.size(0)
#             n += bsz
#             epoch_loss += loss_to_log.item() * bsz
#
#             all_probs.append(probs.detach().cpu())
#             all_targets.append(targets.detach().cpu())
#
#     avg_loss = epoch_loss /  max(1, n)
#     acc = correct /  max(1, n)
#     probs_np = torch.cat(all_probs).numpy().ravel()
#     targets_np = torch.cat(all_targets).numpy().ravel()
#
#     # ROC-AUC
#     try: auc = roc_auc_score(targets_np, probs_np)
#     except Exception: auc = float("nan")
#
#     # PR-AUC (Average Precision)
#     try: pr_auc = average_precision_score(targets_np, probs_np)
#     except Exception: pr_auc = float("nan")
#
#     return avg_loss, acc, auc, pr_auc, probs_np, targets_np


def train_one_epoch(model, loader, epochNumber, optimizer, criterion, device, epochs,
                    w_neg: float = 1.0, w_pos: float = 1.0):
    return _run_epoch(model, loader, criterion, device,
                      is_train=True, optimizer=optimizer,
                      acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Train")
    # return _run_epoch_with_sam(model, loader, criterion, device,
    #                   is_train=True, optimizer=optimizer,
    #                   acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Train",
    #                            w_neg=w_neg, w_pos=w_pos)


def eval_one_epoch(model, loader, epochNumber, criterion, device, epochs,
                   w_neg: float = 1.0, w_pos: float = 1.0):
    return _run_epoch(model, loader, criterion, device,
                      is_train=False, optimizer=None,
                      acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Eval")
    # return _run_epoch_with_sam(model, loader, criterion, device,
    #                   is_train=False, optimizer=None,
    #                   acc_threshold=0.5, epochNumber=epochNumber, epochs=epochs, tag="Eval",
    #                            w_neg=w_neg, w_pos=w_pos)


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
    loss, acc05, auc, pr_auc, probs, targets = eval_one_epoch(model, loader, 1, criterion, device, epochs)
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
    print(f"Specificity: {spec:.4f} | Sensitivity: {sens:.4f} | "f"Balanced Acc: {bal_acc:.4f}")
    print(f"PR-AUC from average_precision_score: {pr_auc:.4f}  | PR-AUC from sklearn.metrics: {pr_auc:.4f}")
    print("Confusion Matrix :\n", cm)
    print("\nClassification Report:\n", classification_report(targets, preds, target_names=CLASS_NAMES, digits=4))

    # ROC curve (test)
    fpr, tpr, _ = roc_curve(targets, probs)
    plot_roc_curve(fpr, tpr)


# # ---------- SAM trick ----------
# class SAM(torch.optim.Optimizer):
#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         assert rho >= 0.0
#         defaults = dict(rho=rho, **kwargs)
#         super().__init__(params, defaults)
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self._skip_second = False
#
#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         rho = self.defaults["rho"]
#         grad_norm = self._grad_norm()
#
#         # If no grads yet, skip SAM ascent; do NOT crash
#         if grad_norm is None or grad_norm.item() == 0.0:
#             self._skip_second = True
#             if zero_grad:
#                 self.zero_grad()
#             return
#
#         scale = rho / (grad_norm + 1e-12)
#         self._skip_second = False
#
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 e = p.grad * scale
#                 p.add_(e)                     # ascent step
#                 self.state[p]["e"] = e
#         if zero_grad:
#             self.zero_grad()
#
#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         # If first_step skipped (no grads), just do a normal step
#         if self._skip_second:
#             self.base_optimizer.step()
#             if zero_grad:
#                 self.zero_grad()
#             self._skip_second = False
#             return
#
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 e = self.state[p].get("e", None)
#                 if e is not None:
#                     p.sub_(e)                 # return to w
#
#         self.base_optimizer.step()
#         if zero_grad:
#             self.zero_grad()
#
#     @torch.no_grad()
#     def step(self):  # not used
#         raise RuntimeError("Call first_step/second_step explicitly.")
#
#     def zero_grad(self, set_to_none: bool = False):   # <-- accept the kwarg
#         self.base_optimizer.zero_grad(set_to_none=set_to_none)
#
#     def _grad_norm(self):
#         norms = []
#         dev = None
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is not None:
#                     g = p.grad.detach()
#                     norms.append(g.norm(p=2))
#                     if dev is None:
#                         dev = g.device
#         if not norms:
#             return torch.tensor(0.0, device=dev or self.param_groups[0]["params"][0].device)
#         return torch.norm(torch.stack(norms), p=2)
