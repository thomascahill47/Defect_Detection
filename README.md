# Defect Detection

Binary image classifier (Accept / Reject) trained on the [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). Uses a fine-tuned ResNet18 backbone with transfer learning, Optuna hyperparameter search, and a full evaluation suite (confusion matrices, classification reports, loss/accuracy curves).


---

## How It Works

Three scripts form the complete pipeline:

1. **`multi_set_data_wrangler.py`** — merges multiple MVTec categories into a single binary `accept/reject` dataset with a 70/15/15 train-val-test split.
2. **`model.py`** — trains a ResNet18 classifier, saves checkpoints, plots, and reports to a timestamped output folder.
3. **`optuna_sweeper.py`** — runs a 30-trial Optuna hyperparameter sweep over learning rate, batch size, and weight decay, with median pruning.

---

## Dataset

This project uses the [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). Download it from the official site — it is free for academic use.

The data wrangler is tested with these three categories: `carpet`, `nut`, `toothbrush`. Any MVTec category (or mix) will work since the folder structure is consistent across the dataset.

**Expected MVTec folder structure per category:**
```
carpet/
├── train/
│   └── good/
└── test/
    ├── good/
    ├── color/
    ├── cut/
    └── ...
```

The wrangler maps `train/good` + `test/good` → **accept**, and all non-good test subfolders → **reject**.

---

## Setup

**Python 3.9+ recommended.**

```bash
git clone https://github.com/thomascahill47/Defect_Detection.git
cd Defect_Detection
pip install -r requirements.txt
```

> **Note on PyTorch:** `requirements.txt` installs CPU PyTorch by default. For GPU support, install PyTorch separately following the [official instructions](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

> **Note on Graphviz:** `model.py` generates a network architecture diagram using Graphviz. This requires the Graphviz system binary in addition to the Python package. Install it via your system package manager (e.g. `brew install graphviz`, `apt install graphviz`) or from [graphviz.org](https://graphviz.org/download/). If it is not installed, the script will skip the diagram and continue normally.

---

## Usage

### Step 1 — Build the dataset

```bash
python code/multi_set_data_wrangler.py ./Final_Set \
    path/to/mvtec/carpet \
    path/to/mvtec/nut \
    path/to/mvtec/toothbrush
```

Output: `./Final_Set/` with `train/`, `val/`, `test/` splits, each containing `accept/` and `reject/` subfolders.

---

### Step 2 — Train

```bash
python code/model.py \
    --data_root ./Final_Set \
    --arch resnet18 \
    --pretrained \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 5 \
    --seed 1337
```

Each run creates a timestamped folder under `./results/` containing:

| File | Description |
|------|-------------|
| `config.json` | Full argument snapshot for reproducibility |
| `train_log.csv` | Per-epoch loss and accuracy |
| `loss_curve.png` | Training vs. validation loss plot |
| `accuracy_curve.png` | Training vs. validation accuracy plot |
| `confusion_matrix_*.png` | Confusion matrices for train, val, and test |
| `report.txt` | Precision, recall, F1 per class |
| `checkpoints/best.pt` | Best model weights by validation accuracy |
| `runtime.txt` | Total training time |

**All CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `./animals` | Path to the dataset |
| `--out_root` | `./results` | Output directory |
| `--arch` | `resnet18` | torchvision model name (e.g. `resnet50`) |
| `--pretrained` | off | Use ImageNet pretrained weights |
| `--epochs` | `10` | Max training epochs |
| `--batch_size` | `32` | Images per batch |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-4` | Adam weight decay |
| `--patience` | `5` | Early stopping patience |
| `--seed` | `1337` | Random seed |
| `--img_size` | `224` | Input image size |
| `--num_workers` | `2` | DataLoader worker threads |

---

### Step 3 — Hyperparameter sweep (optional)

Edit the constants at the top of `optuna_sweeper.py` if needed:

```python
DATA_ROOT   = "./Final_Set"   # path to your built dataset
N_TRIALS    = 30              # number of trials
TIMEOUT_SEC = 60 * 60         # 1-hour cap
```

Then run:

```bash
python code/optuna_sweeper.py
```

Results are stored in `optuna_sweep.db` (SQLite). The sweep tunes learning rate, batch size, and weight decay over 30 trials using median pruning to cut unpromising runs early. The best parameters are printed at the end.

To resume an interrupted sweep, just run the script again — `load_if_exists=True` picks up where it left off.

---

## Results

Results from the best Optuna trial on the carpet + nut + toothbrush split are saved in `results/`.

---

## Project Structure

```
Defect_Detection/
├── code/
│   ├── model.py                  # Training script
│   ├── multi_set_data_wrangler.py # Dataset builder
│   └── optuna_sweeper.py         # Hyperparameter search
├── presentation/                 # Slide deck
├── report/                       # IEEE-format written report
├── results/                      # Sample run outputs
├── requirements.txt
└── optuna_sweep.db               # Optuna study (SQLite)
```

---

## Dependencies

```
torch
torchvision
optuna
scikit-learn
matplotlib
numpy
tqdm
torchviz
graphviz
```

See `requirements.txt` for pinned versions.
