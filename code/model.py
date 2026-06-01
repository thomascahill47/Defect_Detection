#!/usr/bin/env python3

"""
PROFESSIONAL ML WORKBENCH - Image Classification

Updated for CSCI 484 - Step 3 & 4 (Argparse + Output Organization)

This script trains a ResNet18 model on any image dataset organized by folders.
It creates a timestamped output directory for every run containing:
1. Best model checkpoints
2. Loss and Accuracy curves (plots)
3. Confusion Matrices (plots)
4. Raw data (CSV logs, JSON configs)
5. Human-readable reports

Usage:
  python model.py --data_root ./Final_Set --epochs 10 --lr 0.001 --pretrained
"""

import argparse
import os
import random
import sys
import time
import json
import csv
import shutil
from datetime import datetime
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchviz import make_dot
import graphviz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# =============================================================================
# CLASS: WorkbenchOutput
# =============================================================================

class WorkbenchOutput:
    """Handles all file output. Keeps the training loop clean."""

    def __init__(self, base_dir="./results", args=None):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(base_dir, f"run_{self.timestamp}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        print(f"[Workbench] Output directory created: {self.run_dir}")

        if args:
            with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
                json.dump(vars(args), f, indent=4)

        self.csv_path = os.path.join(self.run_dir, "train_log.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def save_checkpoint(self, model, is_best=False):
        torch.save(model.state_dict(), os.path.join(self.run_dir, "model_state_dict.pt"))
        if is_best:
            torch.save(model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))

    def save_curves(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_acc'], label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "accuracy_curve.png"))
        plt.close()

    def save_confusion_matrix(self, y_true, y_pred, class_names, split_name):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {split_name}')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.run_dir, f"confusion_matrix_{split_name}.png"))
        plt.close()

    def save_classification_report(self, y_true, y_pred, class_names, split_name):
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

        with open(os.path.join(self.run_dir, "report.txt"), "a") as f:
            f.write(f"\n{'='*30}\n")
            f.write(f"Results for: {split_name}\n")
            f.write(f"{'='*30}\n")
            f.write(report)
            f.write("\n\n")

    def save_context_file(self, class_names):
        filepath = os.path.join(self.run_dir, "metrics_context.txt")
        with open(filepath, "w") as f:
            f.write("UNDERSTANDING THE METRICS FOR THIS MODEL\n")
            f.write("========================================\n\n")
            f.write(f"Classes: {class_names}\n\n")

            for cls in class_names:
                f.write(f"--- Context for class '{cls}' ---\n")
                f.write(f"True Positive (TP): An image WAS a {cls}, and the model guessed {cls}.\n")
                f.write(f"True Negative (TN): An image was NOT a {cls}, and the model correctly guessed something else.\n")
                f.write(f"False Positive (FP): The model guessed {cls}, but it was actually something else (False Alarm).\n")
                f.write(f"False Negative (FN): The image WAS a {cls}, but the model missed it (Missed Detection).\n\n")

    def save_runtime(self, start_time):
        end_time = time.time()
        duration_sec = end_time - start_time
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        seconds = int(duration_sec % 60)

        with open(os.path.join(self.run_dir, "runtime.txt"), "w") as f:
            f.write(f"Total Training Time: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Start Timestamp: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"End Timestamp: {datetime.now()}\n")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(data_root, batch_size, image_size=224, num_workers=2):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=eval_tf)
        test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=eval_tf)
    except FileNotFoundError:
        print(f"ERROR: Could not find data folders at {data_root}.")
        print("Expected structure: data_root/train, data_root/val, data_root/test")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes, arch="resnet18", pretrained=True):
    if not hasattr(models, arch):
        print(f"Error: {arch} not found in torchvision.models. Defaulting to resnet18.")
        arch = "resnet18"

    model_func = getattr(models, arch)
    model = model_func(pretrained=pretrained)

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        print("Warning: Model head is not 'fc'. Check architecture compatibility.")

    return model


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch_idx):
    model.train()
    running_loss, correct_preds, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Batches e{epoch_idx}", leave=False, unit="batch", colour='cyan')

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / total_samples, correct_preds / total_samples


@torch.no_grad()
def evaluate_dataset(model, loader, loss_fn, device):
    model.eval()

    running_loss, correct_preds, total_samples = 0.0, 0, 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return running_loss / total_samples, correct_preds / total_samples, y_true, y_pred


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Defect Detection - ResNet Image Classifier")

    parser.add_argument("--data_root",     type=str,   default="./Final_Set", help="Path to dataset")
    parser.add_argument("--out_root",      type=str,   default="./results",   help="Path to output results")
    parser.add_argument("--img_size",      type=int,   default=224,           help="Image input size")
    parser.add_argument("--num_workers",   type=int,   default=2,             help="DataLoader worker threads")
    parser.add_argument("--seed",          type=int,   default=1337,          help="Random seed")
    parser.add_argument("--arch",          type=str,   default="resnet18",    help="Model architecture (e.g. resnet18, resnet50)")
    parser.add_argument("--pretrained",    action='store_true',               help="Use ImageNet pretrained weights")
    parser.add_argument("--epochs",        type=int,   default=10,            help="Max training epochs")
    parser.add_argument("--batch_size",    type=int,   default=32,            help="Images per batch")
    parser.add_argument("--lr",            type=float, default=1e-3,          help="Learning rate")
    parser.add_argument("--weight_decay",  type=float, default=1e-4,          help="Adam weight decay")
    parser.add_argument("--patience",      type=int,   default=5,             help="Early stopping patience")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wb = WorkbenchOutput(base_dir=args.out_root, args=args)
    start_time = time.time()

    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading data from {args.data_root}...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root, args.batch_size,
        image_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f"Classes: {class_names}")
    wb.save_context_file(class_names)

    print(f"Building {args.arch} model (pretrained={args.pretrained})...")
    model = build_model(len(class_names), arch=args.arch, pretrained=args.pretrained)
    model = model.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Network architecture diagram (requires graphviz system binary)
    print("\nGenerating network diagram...")
    try:
        graph_dir = os.path.join(wb.run_dir, 'graphviz')
        os.makedirs(graph_dir, exist_ok=True)

        dummy_input, _ = next(iter(train_loader))
        dummy_input = dummy_input.to(device)

        y_hat      = model(dummy_input)
        activation = torch.nn.Softmax(dim=1)
        output     = activation(y_hat)

        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E0E0E0', fontname='Arial')

        output_path = os.path.join(graph_dir, 'model_flowchart')
        dot.format = 'png'
        dot.render(output_path, engine='dot')
        print(f"Graph saved to: {output_path}.png")
    except Exception as e:
        print(f"Warning: Graphviz diagram skipped. ({e})")

    print(f"\033[1m\033[96m{args.arch.upper()} TRAINING DASHBOARD\033[0m")
    print(f"Output: \033[92m{wb.run_dir}\033[0m | Device: \033[93m{device}\033[0m")

    best_val_acc    = 0.0
    epochs_no_improve = 0

    pbar_epochs = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch", colour='cyan')

    for epoch in pbar_epochs:
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        v_loss, v_acc, _, _ = evaluate_dataset(model, val_loader, loss_fn, device)

        wb.log_epoch(epoch, t_loss, t_acc, v_loss, v_acc)

        # Hook for optuna_sweeper.py to read validation loss
        print(f"OPTUNA_VAL_LOSS: {v_loss:.4f} EPOCH: {epoch}")
        sys.stdout.flush()

        pbar_epochs.set_postfix({
            'valLoss': f"{v_loss:.4f}",
            'valAcc':  f"{v_acc:.4f}",
            'best':    f"{best_val_acc:.4f}"
        })

        tqdm.write(f" \033[93m★\033[0m epoch {epoch:3d} | train Loss {t_loss:.4f} Acc {t_acc:.4f} | val Loss {v_loss:.4f} Acc {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            epochs_no_improve = 0
            tqdm.write("   --> \033[92mNew best!\033[0m Saving checkpoint.")
            wb.save_checkpoint(model, is_best=True)
        else:
            epochs_no_improve += 1
            tqdm.write(f"   --> No improvement. Patience: {epochs_no_improve}/{args.patience}")
            wb.save_checkpoint(model, is_best=False)

        if epochs_no_improve >= args.patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print("\nTraining complete. Generating reports...")
    wb.save_curves()

    best_model_path = os.path.join(wb.ckpt_dir, "best.pt")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    for split_name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        print(f"   {split_name} metrics...")
        _, _, true_labels, pred_labels = evaluate_dataset(model, loader, loss_fn, device)
        wb.save_confusion_matrix(true_labels, pred_labels, class_names, split_name)
        wb.save_classification_report(true_labels, pred_labels, class_names, split_name)

    wb.save_runtime(start_time)

    print("-" * 60)
    print(f"Done. Results saved in: {wb.run_dir}")
    print("-" * 60)


if __name__ == "__main__":
    main()
