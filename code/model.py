#!/usr/bin/env python3

"""
# This one has valid ass results output folder


PROFESSIONAL ML WORKBENCH - Image Classification

------------------------------------------------

Updated for CSCI 484 - Step 3 & 4 (Argparse + Output Organization)



This script trains a ResNet18 model on any image dataset organized by folders.

It creates a timestamped output directory for every run containing:

1. Best model checkpoints

2. Loss and Accuracy curves (plots)

3. Confusion Matrices (plots)

4. Raw data (CSV logs, JSON configs)

5. Human-readable reports



Usage:

  python animals_workbench.py --data_root ./animals --epochs 10 --lr 0.001

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
import graphviz

# Math and ML libraries

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms
from torchviz import make_dot
import graphviz



# Reporting libraries

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report





# =============================================================================

# CLASS: WorkbenchOutput (The "Accountant")

# =============================================================================

class WorkbenchOutput:

    """

    This class handles ALL file output. It keeps the training loop clean.

    It creates the folder structure and saves charts, logs, and models.

    """

    def __init__(self, base_dir="./results", args=None):

        # 1. Create a unique timestamped folder for this run

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_dir = os.path.join(base_dir, f"run_{self.timestamp}")

        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")

       

        # Create directories

        os.makedirs(self.run_dir, exist_ok=True)

        os.makedirs(self.ckpt_dir, exist_ok=True)

       

        print(f"[Workbench] Output directory created: {self.run_dir}")



        # 2. Save the Config (Recipe) immediately

        if args:

            with open(os.path.join(self.run_dir, "config.json"), 'w') as f:

                json.dump(vars(args), f, indent=4)



        # 3. Initialize the CSV Log (The Spreadsheet)

        self.csv_path = os.path.join(self.run_dir, "train_log.csv")

        with open(self.csv_path, 'w', newline='') as f:

            writer = csv.writer(f)

            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])



        # Storage for plotting later

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}



    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):

        """Append a single epoch's stats to the CSV and memory."""

        # Write to CSV

        with open(self.csv_path, 'a', newline='') as f:

            writer = csv.writer(f)

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

       

        # Store in memory for plotting

        self.history['train_loss'].append(train_loss)

        self.history['train_acc'].append(train_acc)

        self.history['val_loss'].append(val_loss)

        self.history['val_acc'].append(val_acc)



    def save_checkpoint(self, model, is_best=False):

        """Save the model weights."""

        # Always save the 'latest' state (optional, strictly requested 'best')

        # specific request: "model_state_dict.pt" as the final readable one

        torch.save(model.state_dict(), os.path.join(self.run_dir, "model_state_dict.pt"))

       

        if is_best:

            torch.save(model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))



    def save_curves(self):

        """Draws the colorful lines for Loss and Accuracy."""

        epochs = range(1, len(self.history['train_loss']) + 1)



        # Plot Loss

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



        # Plot Accuracy

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

        """Generates a colorful Confusion Matrix PNG."""

        cm = confusion_matrix(y_true, y_pred)

       

        plt.figure(figsize=(8, 6))

        # Use simple matplotlib imshow to avoid needing seaborn dependency

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        plt.title(f'Confusion Matrix - {split_name}')

        plt.colorbar()

       

        tick_marks = np.arange(len(class_names))

        plt.xticks(tick_marks, class_names, rotation=45)

        plt.yticks(tick_marks, class_names)



        # Add text annotations inside the squares

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

        """Saves the text report (Precision/Recall/F1)."""

        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

       

        # We append to a main report file

        with open(os.path.join(self.run_dir, "report.txt"), "a") as f:

            f.write(f"\n{'='*30}\n")

            f.write(f"Results for: {split_name}\n")

            f.write(f"{'='*30}\n")

            f.write(report)

            f.write("\n\n")



    def save_context_file(self, class_names):

        """Creates a text file explaining TP/FP/TN/FN for these specific classes."""

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

        """Saves how long the training took."""

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

# HELPER FUNCTIONS (Device, Data, Model)

# =============================================================================



def pick_device():

    """Step 1: Check if we have a GPU (MPS or CUDA) or use CPU."""

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")



def build_dataloaders(data_root, batch_size, image_size=224, num_workers = 2):

    """

    Step 2: Create the data pipelines.

    - Train: Augmentation (Random flips) to make the model robust.

    - Val/Test: No augmentation, just resize.

    """

    # Define Transforms

    train_tf = transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(p=0.5), # Augmentation

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])



    eval_tf = transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])



    # Load Folders

    try:

        train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)

        val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf)

        test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=eval_tf)

    except FileNotFoundError as e:

        print(f"ERROR: Could not find data folders at {data_root}.")

        print("Expected structure: data_root/train, data_root/val, data_root/test")

        sys.exit(1)



    # Wrap in Loaders

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers= num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers= num_workers)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers= num_workers)



    return train_loader, val_loader, test_loader, train_ds.classes



def build_model(num_classes, arch="resnet18", pretrained=True):
    """Step 3: Build the Neural Network."""
    # Dynamically load the model from torchvision.models
    if not hasattr(models, arch):
        print(f"Error: {arch} not found in torchvision.models. Defaulting to resnet18.")
        arch = "resnet18"
    
    # Fetch the architecture function (e.g., models.resnet50)
    model_func = getattr(models, arch)
    model = model_func(pretrained=pretrained)
    
    # Replace the "head". Note: This assumes ResNet-style architecture (model.fc)
    # If using VGG/DenseNet, the head layer name changes (e.g., model.classifier)
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        print("Warning: Model architecture head not strictly 'fc'. Check implementation.")
    
    return model



# =============================================================================

# THE BRAIN: Training and Evaluation Loops

# =============================================================================



def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch_idx): # Added epoch_idx
    model.train()
    running_loss, correct_preds, total_samples = 0.0, 0, 0

    # The 'leave=False' makes this bar disappear when the epoch finishes
    pbar_batches = tqdm(loader, desc=f"Batches e{epoch_idx}", leave=False, unit="batch", colour='cyan')

    for images, labels in pbar_batches:
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Update the mini batch bar's loss display
        pbar_batches.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / total_samples, correct_preds / total_samples



@torch.no_grad() # Disable gradient calculation for speed (we aren't learning here)

def evaluate_dataset(model, loader, loss_fn, device):

    """

    Runs the model on a dataset without learning.

    Used for Validation and Testing.

    Returns: Loss, Accuracy, True Labels List, Predicted Labels List

    """

    model.eval() # Set to evaluation mode

   

    running_loss = 0.0

    correct_preds = 0

    total_samples = 0

   

    # Lists to store raw results for Confusion Matrix

    y_true = []

    y_pred = []



    for images, labels in loader:

        images, labels = images.to(device), labels.to(device)



        # Forward Pass only

        logits = model(images)

        loss = loss_fn(logits, labels)



        # Record Keeping

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        correct_preds += (preds == labels).sum().item()

        total_samples += labels.size(0)



        # Store for reporting

        y_true.extend(labels.cpu().tolist())

        y_pred.extend(preds.cpu().tolist())



    avg_loss = running_loss / total_samples

    accuracy = correct_preds / total_samples

    return avg_loss, accuracy, y_true, y_pred





# =============================================================================

# MAIN EXECUTION

# =============================================================================



def main():

    # --- 1. ARGUMENT PARSING (The Knobs & Dials) ---
    parser = argparse.ArgumentParser(description="Animals Workbench - Pro CLI")
    
    # Paths
    parser.add_argument("--data_root", type=str, default="./animals", help="Path to input dataset")
    parser.add_argument("--out_root", type=str, default="./results", help="Path to output results")
    
    # Data Config
    parser.add_argument("--img_size", type=int, default=224, help="Image input size")
    parser.add_argument("--num_workers", type=int, default=2, help="CPU workers for data loading")
    # Note: These split ratios are for logging; this script assumes folders are pre-split
    parser.add_argument("--train", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Val split ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")

    # Architecture
    parser.add_argument("--arch", type=str, default="resnet18", help="Model architecture (resnet18, resnet50)")
    parser.add_argument("--pretrained", action='store_true', help="Use pretrained weights")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training loops")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()

    # --- 1.5 APPLY SEED (New Step) ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)



    # --- 2. SETUP WORKBENCH ---

    # Initialize our "Accountant" to handle file saving

    wb = WorkbenchOutput(base_dir=args.out_root, args=args)

    start_time = time.time()



    # Setup Device (GPU/CPU)

    device = pick_device()

    print(f" Device selected: {device}")




    # --- 3. LOAD DATA ---

    print(f" Loading data from {args.data_root}...")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root, 
        args.batch_size, 
        image_size=args.img_size,
        num_workers=args.num_workers 
    )

    print(f"   Classes found: {class_names}")

   

    # Save the Context File (Explanation of TP/FP/etc)

    wb.save_context_file(class_names)



    # --- 4. BUILD MODEL ---

    print("Building ResNet18 model...")

    print(f"Building {args.arch} model...")
    model = build_model(len(class_names), arch=args.arch, pretrained=args.pretrained)

    model = model.to(device)



    # --- 5. SETUP TRAINING TOOLS ---

    loss_fn = nn.CrossEntropyLoss()
    # Added weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------------------------------------------------
    # GRAPHVIZ: Generate Network Visualization
    # ---------------------------------------------------------
    print("\nGenerating educational network visualization...")
    
    # WINDOWS FIX: Ensure Python can find the Graphviz 'dot' command
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    try:
        # 1. Create the output directory
        graph_dir = os.path.join(wb.run_dir, 'graphviz')
        os.makedirs(graph_dir, exist_ok=True)

        # 2. Get dummy input (1 image batch)
        dummy_input, _ = next(iter(train_loader)) 
        dummy_input = dummy_input.to(device)

        # 3. THE "STUDENT" FORWARD PASS
        # We break this down step-by-step to match ML vocabulary
        y_hat = model(dummy_input)             # Step A: Raw Output (Logits)
        activation = torch.nn.Softmax(dim=1)   # Step B: Pick Activation Function
        output = activation(y_hat)             # Step C: Final Probabilities

        # 4. Generate the graph object
        # We trace 'output' so the graph includes the Activation Function node
        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)

        # 5. CUSTOMIZE: Add the "Lesson" Legend
        # This adds the text explaining the math at the top of the image
        dot.attr(rankdir='LR') # Left-to-Right layout (Timeline view)
        
        legend_text = (
            r'\nNEURAL NETWORK FLOWCHART\n'
            r'----------------------------------------\n'
            r'1. INPUT: Image pixels enter on the Left.\n'
            r'2. HIDDEN LAYERS: The boxes transform features (Edges -> Shapes -> Objects).\n'
            r'3. y_hat (LOGITS): The raw scores come out of the last blue box.\n'
            r'4. ACTIVATION FUNCTION: Softmax converts y_hat into percentages.\n'
            r'5. OUTPUT: The final node is the Probability Distribution.\n'
            r'   (Example: [0.10, 0.90] means 90% Confidence -> PREDICT: CLASS 1)'
        )
        
        # Apply the text and styling
        dot.attr(label=legend_text, labelloc='t', fontsize='14', fontname='Arial')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='#E0E0E0', fontname='Arial')

        # 6. Save and render
        output_path = os.path.join(graph_dir, 'model_flowchart')
        dot.format = 'png'
        dot.render(output_path, engine='dot')
        
        print(f" Educational graph saved to: {output_path}.png")

    except Exception as e:
        print(f" Warning: Graphviz failed. Error: {e}")
    

    # --- 6. TRAINING LOOP ---

    print(f"\033[1m\033[96mIMAGE CLASSIFICATION - {args.arch.upper()} TRAINING DASHBOARD\033[0m")
    print(f"Output dir: \033[92m{wb.run_dir}\033[0m")
    print(f"Device: \033[93m{device}\033[0m")
    
    best_val_acc = 0.0
    epochs_no_improve = 0  # Counter for patience



    # --- 6. TRAINING LOOP (WITH DASHBOARD) ---
    pbar_epochs = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch", colour='cyan')

    for epoch in pbar_epochs:
        # A. Train (Now passing 'epoch' for the nested batch bar)
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)

        # B. Validate
        v_loss, v_acc, _, _ = evaluate_dataset(model, val_loader, loss_fn, device)

        # C. Log to CSV and History (KEEP THIS)
        wb.log_epoch(epoch, t_loss, t_acc, v_loss, v_acc)
        # --- OPTUNA HOOK START ---
        # This allows the sweeper to read the loss in real-time
        print(f"OPTUNA_VAL_LOSS: {v_loss:.4f} EPOCH: {epoch}")
        sys.stdout.flush() # Force Python to push the text out immediately
        # --- OPTUNA HOOK END ---

        # D. Dashboard Update
        # This updates the text on the right side of the blue bar
        pbar_epochs.set_postfix({
            'valLoss': f"{v_loss:.4f}",
            'valAcc': f"{v_acc:.4f}",
            'best': f"{best_val_acc:.4f}"
        })

        # This prints the yellow star line without breaking the progress bar
        tqdm.write(f" \033[93m★\033[0m epoch {epoch:3d} | train Loss {t_loss:.4f} Acc {t_acc:.4f} | val Loss {v_loss:.4f} Acc {v_acc:.4f}")

        # E. Checkpointing & Early Stopping (KEEP THIS)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            epochs_no_improve = 0
            tqdm.write(f"   --> \033[92mNew Best Score!\033[0m Saving checkpoint.")
            wb.save_checkpoint(model, is_best=True)
        else:
            epochs_no_improve += 1
            tqdm.write(f"   --> No improvement. Patience: {epochs_no_improve}/{args.patience}")
            wb.save_checkpoint(model, is_best=False)

        if epochs_no_improve >= args.patience:
            tqdm.write(f"\n Early stopping triggered after {epoch} epochs.")
            break

    # --- 7. FINAL REPORTING ---

    print("\n Training Complete. Generating Reports...")

   

    # Save the Loss/Accuracy Curves

    wb.save_curves()



    # Reload the Best Model for final evaluation

    best_model_path = os.path.join(wb.ckpt_dir, "best.pt")

    model.load_state_dict(torch.load(best_model_path))

    model.to(device)



    # Generate Reports for ALL splits (Train, Val, Test)

    # We iterate through them to avoid repeating code

    splits = [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]

   

    for split_name, loader in splits:

        print(f"   Processing {split_name} metrics...")

        _, _, true_labels, pred_labels = evaluate_dataset(model, loader, loss_fn, device)

       

        # Save Confusion Matrix

        wb.save_confusion_matrix(true_labels, pred_labels, class_names, split_name)

       

        # Save Classification Report (Precision/Recall/F1)

        wb.save_classification_report(true_labels, pred_labels, class_names, split_name)



    # Save Runtime

    wb.save_runtime(start_time)



    print("-" * 60)

    print(f"Done! All results saved in: {wb.run_dir}")

    print("-" * 60)



if __name__ == "__main__":

    main() 