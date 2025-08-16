import os, random
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import torch
import shutil
import tempfile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

TRAIN_VAL = ["train", "val"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
KAGGLE_DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_xray_dataset():
    path = kagglehub.dataset_download(KAGGLE_DATASET_NAME)
    base_dir = os.path.join(path, "chest_xray")

    train_dir = os.path.join(base_dir, 'train')
    print("Train dir:", train_dir)
    print("Train dir contents:", os.listdir(train_dir))

    val_dir = os.path.join(base_dir, 'val')
    print("Val dir:", val_dir)
    print("Val dir contents:", os.listdir(val_dir))

    test_dir = os.path.join(base_dir, 'test')
    print("Test dir:", test_dir)
    print("Test dir contents:", os.listdir(test_dir))

    return base_dir, train_dir, val_dir, test_dir


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


def show_split_counts(base_dir, title):
    print(f"{title}:")
    for split in TRAIN_VAL:
        for cls in CLASS_NAMES:
            folder = os.path.join(base_dir, split, cls)
            print(f"{split}/{cls}: {len(os.listdir(folder))}")
    plot_distribution_from_folder(base_dir, title)


def balance_val_train_split(base_dir):
    show_split_counts(base_dir, "Before split")

    all_data = []
    for split in TRAIN_VAL:
        for cls in CLASS_NAMES:
            folder = os.path.join(base_dir, split, cls)
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

    show_split_counts(new_base, "After split")
    print("New dir:", new_base)

    return new_base


# ---------- transforms & dataloader helpers ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])


def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers, img_size):
    train_tfms = get_train_transforms(img_size)
    eval_tfms = get_eval_transform(img_size)
    pin = torch.cuda.is_available()

    data_sets = {
        'train_ds' : datasets.ImageFolder(train_dir, transform=train_tfms),
        'val_ds' : datasets.ImageFolder(val_dir, transform=eval_tfms),
        'test_ds' : datasets.ImageFolder(test_dir, transform=eval_tfms)}

    loaders = {
        'train_loader' : DataLoader(data_sets['train_ds'], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin),
        'val_loader' : DataLoader(data_sets['val_ds'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        'test_loader' : DataLoader(data_sets['test_ds'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)}

    return data_sets, loaders


# ---------- plots ----------
def plot_distribution_from_folder(base_dir, title_prefix):
    for split in TRAIN_VAL:
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