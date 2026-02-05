import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description="Merge 3 datasets into one unified Accept/Reject dataset.")

parser.add_argument("output_root", help="Path to output folder")

parser.add_argument("source_folders", nargs='+', help="List of source folders (e.g. path/to/carpet path/to/nut path/to/toothbrush)")
args = parser.parse_args()

output_root = args.output_root
source_folders = args.source_folders

train_frac = 0.7
val_frac = 0.15
test_frac = 0.15

for split in ["train", "val", "test"]:
    for label in ["accept", "reject"]:
        folder = os.path.join(output_root, split, label)
        os.makedirs(folder, exist_ok=True)

def get_images_from_source(source_path):
    """
    Extracts accept/reject images from a specific dataset folder 
    (e.g., Carpet) based on the specific logic:
    - Accept: train/good + test/good
    - Reject: test/* (excluding 'good')
    - Ignored: ground_truth
    """
    local_accepts = []
    local_rejects = []
    
    print(f"Scanning source: {source_path}...")

    # 1. Get Accept images from train/good
    train_good_dir = os.path.join(source_path, "train", "good")
    if os.path.exists(train_good_dir):
        local_accepts += [os.path.join(train_good_dir, f) for f in os.listdir(train_good_dir) 
                          if os.path.isfile(os.path.join(train_good_dir, f))]

    # 2. Get Accept images from test/good
    test_good_dir = os.path.join(source_path, "test", "good")
    if os.path.exists(test_good_dir):
        local_accepts += [os.path.join(test_good_dir, f) for f in os.listdir(test_good_dir) 
                          if os.path.isfile(os.path.join(test_good_dir, f))]

    # 3. Get Reject images from test/* (Excluding "good")
    test_dir = os.path.join(source_path, "test")
    if os.path.exists(test_dir):
        # Get all subfolders in 'test', but SKIP the folder named 'good'
        reject_subdirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir)
                          if os.path.isdir(os.path.join(test_dir, d)) and d != "good"]

        for d in reject_subdirs:
            local_rejects += [os.path.join(d, f) for f in os.listdir(d) 
                              if os.path.isfile(os.path.join(d, f))]
    
    print(f"   -> Found {len(local_accepts)} Accept | {len(local_rejects)} Reject")
    return local_accepts, local_rejects

master_accept_files = []
master_reject_files = []

print("--- Starting Extraction ---")

for source in source_folders:
    if os.path.exists(source):
        acc, rej = get_images_from_source(source)
        master_accept_files.extend(acc)
        master_reject_files.extend(rej)
    else:
        print(f"WARNING: Source folder not found: {source}")

print("--- Extraction Complete ---")
print(f"Total Combined Accept: {len(master_accept_files)}")
print(f"Total Combined Reject: {len(master_reject_files)}")

# Shuffle the combined lists
random.shuffle(master_accept_files)
random.shuffle(master_reject_files)

# ------------------------
# Splitting Logic
# ------------------------
def split_files(file_list):
    n = len(file_list)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    # Remaining goes to test to ensure total equals n
    return file_list[:n_train], file_list[n_train:n_train+n_val], file_list[n_train+n_val:]

accept_train, accept_val, accept_test = split_files(master_accept_files)
reject_train, reject_val, reject_test = split_files(master_reject_files)

# ------------------------
# Copy files safely (Rename on collision)
# ------------------------
def copy_files_safe(file_list, split, label):
    count = 0
    for src in file_list:
        base = os.path.basename(src)
        dst = os.path.join(output_root, split, label, base)
        
        # If file exists (collision from different datasets), rename it
        counter = 1
        while os.path.exists(dst):
            name, ext = os.path.splitext(base)
            dst = os.path.join(output_root, split, label, f"{name}_{counter}{ext}")
            counter += 1
            
        try:
            shutil.copy2(src, dst)
            count += 1
        except PermissionError:
            print(f"Permission denied, skipping: {dst}")
    return count

print("--- Copying Files ---")

# Copy accept images
a_train_c = copy_files_safe(accept_train, "train", "accept")
a_val_c   = copy_files_safe(accept_val, "val", "accept")
a_test_c  = copy_files_safe(accept_test, "test", "accept")

# Copy reject images
r_train_c = copy_files_safe(reject_train, "train", "reject")
r_val_c   = copy_files_safe(reject_val, "val", "reject")
r_test_c  = copy_files_safe(reject_test, "test", "reject")

# ------------------------
# Summary
# ------------------------
print("--- Done! ---")
print(f"Train set:      Accept={a_train_c}, Reject={r_train_c}")
print(f"Validation set: Accept={a_val_c},   Reject={r_val_c}")
print(f"Test set:       Accept={a_test_c},  Reject={r_test_c}")
print(f"Unified dataset ready at: {output_root}")