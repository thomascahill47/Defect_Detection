import optuna
import subprocess
import sys
import os
import re

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_SCRIPT = "model.py"
DATA_ROOT = "./Final_Set"  # Update this to your path
N_TRIALS = 30              # How many attempts to make
TIMEOUT_SEC = 60 * 60      # 1 Hour limit (optional)
STORAGE_DB = "sqlite:///optuna_sweep.db"

def objective(trial):
    # 1. Suggest Parameters (UPDATED RANGES)
    
    # Narrowed LR: 0.0001 to 0.005. 
    # We cap it at 0.005 because 0.01 is often unstable for ResNets on small data.
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    # User requested: Smaller batches for better generalization on small data
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Narrowed Weight Decay: 0.0001 to 0.01
    # We increased the lower bound. You NEED decay here to prevent overfitting.
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    
    arch = "resnet18" 

    # 2. Setup output path
    run_name = f"optuna_trial_{trial.number}"
    out_dir = os.path.join("optuna_results", run_name)

    # 3. Construct Command
    # Note: We added "--pretrained" flag conditionally? 
    # Actually, for small data, you usually SHOULD use pretrained. 
    # If you want to force it, uncomment the line below:
    # cmd_flags = ["--pretrained"] 
    
    cmd = [
        sys.executable, "-u", TRAIN_SCRIPT,
        "--data_root", DATA_ROOT,
        "--out_root", out_dir,
        "--epochs", "10", 
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--weight_decay", str(weight_decay),
        "--arch", arch,
        "--patience", "3"
    ]
    
    # Add pretrained if you decide to use it (Highly Recommended for <1000 images)
    # cmd.append("--pretrained") 

    print(f"\n[Trial {trial.number}] Starting: LR={lr:.4f}, BS={batch_size}, WD={weight_decay:.4f}")

    # 4. Run Process & Monitor Output for Pruning
    best_val_loss_this_run = float('inf')
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1) as proc:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            
            if line:
                # Look for the tag: "OPTUNA_VAL_LOSS: 0.1234 EPOCH: 1"
                match = re.search(r"OPTUNA_VAL_LOSS:\s+([0-9\.]+)\s+EPOCH:\s+(\d+)", line)
                
                if match:
                    val_loss = float(match.group(1))
                    epoch = int(match.group(2))
                    
                    if val_loss < best_val_loss_this_run:
                        best_val_loss_this_run = val_loss

                    # REPORT to Optuna
                    trial.report(val_loss, epoch)

                    # PRUNING CHECK
                    if trial.should_prune():
                        print(f"✂️  Pruning Trial {trial.number} at Epoch {epoch} (Loss: {val_loss:.4f} is too high)")
                        proc.kill()
                        raise optuna.TrialPruned()

    if best_val_loss_this_run == float('inf'):
        return float('inf')
        
    return best_val_loss_this_run

if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(
        direction="minimize", 
        storage=STORAGE_DB, 
        study_name="resnet_optimization",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    print("🚀 Starting Optuna Sweep...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC)
    except KeyboardInterrupt:
        print("Stopping...")

    print("="*50)
    print("🏆 BEST PARAMS FOUND")
    print(study.best_params)
    print(f"Best Loss: {study.best_value}")
    print("="*50)