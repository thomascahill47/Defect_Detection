import optuna
import subprocess
import sys
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_SCRIPT = "model.py"
DATA_ROOT    = "./Final_Set"   # Update to your dataset path
N_TRIALS     = 30
TIMEOUT_SEC  = 60 * 60         # 1-hour cap
STORAGE_DB   = "sqlite:///optuna_sweep.db"


def objective(trial):
    lr           = trial.suggest_float("lr",           1e-4, 5e-3,  log=True)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2,  log=True)
    arch         = "resnet18"

    out_dir  = os.path.join("optuna_results", f"trial_{trial.number}")

    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--data_root",    DATA_ROOT,
        "--out_root",     out_dir,
        "--epochs",       "10",
        "--lr",           str(lr),
        "--batch_size",   str(batch_size),
        "--weight_decay", str(weight_decay),
        "--arch",         arch,
        "--patience",     "3",
        "--pretrained",
    ]

    print(f"\n[Trial {trial.number}] LR={lr:.4f}  BS={batch_size}  WD={weight_decay:.4f}")

    best_val_loss = float('inf')

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1) as proc:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break

            if line:
                match = re.search(r"OPTUNA_VAL_LOSS:\s+([0-9\.]+)\s+EPOCH:\s+(\d+)", line)
                if match:
                    val_loss = float(match.group(1))
                    epoch    = int(match.group(2))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    trial.report(val_loss, epoch)

                    if trial.should_prune():
                        print(f"Pruning trial {trial.number} at epoch {epoch} (loss {val_loss:.4f})")
                        proc.kill()
                        raise optuna.TrialPruned()

    return best_val_loss if best_val_loss != float('inf') else float('inf')


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        storage=STORAGE_DB,
        study_name="resnet_optimization",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    print("Starting Optuna sweep...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC)
    except KeyboardInterrupt:
        print("Stopped early.")

    print("=" * 50)
    print("BEST PARAMS FOUND")
    print(study.best_params)
    print(f"Best loss: {study.best_value:.4f}")
    print("=" * 50)
