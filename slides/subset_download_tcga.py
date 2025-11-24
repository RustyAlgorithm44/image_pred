#!/usr/bin/env python3
import os
import shutil
import subprocess
import pandas as pd
# ---------- CONFIG ----------
MANIFEST_DIR = "processed_manifests"  # folder with manifests
OUT_DIR = "downloaded_svs"            # where downloads go
GDC_CLIENT = "gdc-client"             # must be in PATH
# ----------------------------

def read_manifest(path):
    """Reads a tab-separated manifest file as a DataFrame."""
    return pd.read_csv(path, sep="\t")

def subset_manifest(df, n, mode="random"):
    """Returns a subset (random or sequential)."""
    if n > len(df):
        n = len(df)
    if mode == "random":
        return df.sample(n, random_state=42)
    return df.head(n)

def write_temp_manifest(df, name):
    """Writes a subset manifest to a temp file."""
    temp_path = f"temp_manifest_{name}.txt"
    df.to_csv(temp_path, sep="\t", index=False)
    return temp_path

def gdc_download(manifest_path, outdir):
    """Runs gdc-client to download files into a subdirectory."""
    cmd = [GDC_CLIENT, "download", "-m", manifest_path, "-d", outdir]
    subprocess.run(cmd, check=True)

def flatten_download_dir(download_dir):
    """Moves all .svs files up one level (GDC puts them in subfolders)."""
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if f.endswith(".svs"):
                full_path = os.path.join(root, f)
                target_path = os.path.join(download_dir, f)
                if full_path != target_path:
                    shutil.move(full_path, target_path)

    # remove leftover empty subfolders
    for d in os.listdir(download_dir):
        sub = os.path.join(download_dir, d)
        if os.path.isdir(sub) and not os.listdir(sub):
            os.rmdir(sub)

def rename_files(download_dir, cancer_type):
    """Rename files like TCGA-XX-XXXX_cancerType.svs (only local files)."""
    for f in os.listdir(download_dir):
        if f.endswith(".svs"):
            parts = f.split(".")
            tcga_id = parts[0].split("-")
            tcga_base = "-".join(tcga_id[:3]) if len(tcga_id) >= 3 else parts[0]
            new_name = f"{tcga_base}_{cancer_type}.svs"
            old_path = os.path.join(download_dir, f)
            new_path = os.path.join(download_dir, new_name)
            os.rename(old_path, new_path)

def organize_by_type(download_dir):
    """Create adeno/ and squamous/ folders and move files."""
    adeno_dir = os.path.join(download_dir, "adeno")
    squam_dir = os.path.join(download_dir, "squamous")
    os.makedirs(adeno_dir, exist_ok=True)
    os.makedirs(squam_dir, exist_ok=True)

    for f in os.listdir(download_dir):
        if not f.endswith(".svs"):
            continue
        src = os.path.join(download_dir, f)
        if "squam" in f.lower():
            shutil.move(src, squam_dir)
        elif any(x in f.lower() for x in ["adeno", "coad", "read", "stad", "esca"]):
            shutil.move(src, adeno_dir)

def filter_by_status(df, status_choice):
    """Filter the dataframe by WWOX_del or WWOX_not_del if specified."""
    last_col = df.columns[-1]

    if status_choice == "del":
        return df[df[last_col].str.contains("_del", case=False, na=False) &
                  ~df[last_col].str.contains("not_del", case=False, na=False)]
    elif status_choice == "not_del":
        return df[df[last_col].str.contains("not_del", case=False, na=False)]
    return df

def process_manifest(manifest_file, n_samples, mode, status_choice, out_root):
    """Helper to download, rename, and move one manifest."""
    df = read_manifest(os.path.join(MANIFEST_DIR, manifest_file))
    df = filter_by_status(df, status_choice)
    if df.empty:
        print(f"⚠️ No samples matching '{status_choice}' in {manifest_file}, skipping.")
        return

    sub = subset_manifest(df, n_samples, mode)
    manifest_tag = manifest_file.split(".")[0]
    temp_path = write_temp_manifest(sub, manifest_tag)

    temp_out = os.path.join(out_root, manifest_tag)
    os.makedirs(temp_out, exist_ok=True)

    print(f"\n📥 Downloading {manifest_tag} subset ({len(sub)} samples)...")
    gdc_download(temp_path, temp_out)
    flatten_download_dir(temp_out)
    rename_files(temp_out, manifest_tag)

    # Move renamed files up to OUT_DIR
    for f in os.listdir(temp_out):
        if f.endswith(".svs"):
            shutil.move(os.path.join(temp_out, f), os.path.join(out_root, f))
    shutil.rmtree(temp_out)

    os.remove(temp_path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_squam = int(input("Enter number of squamous samples: "))
    n_adeno_total = int(input("Enter number of adeno samples (total across adeno types): "))
    mode_input = input("Subset mode? ('r' = random, 's' = sequential): ").strip().lower()
    mode = "random" if mode_input == "r" else "sequential"
    status_choice = input("Subset which status? ('del', 'not_del', or 'all'): ").strip().lower()

    squam_files = [f for f in os.listdir(MANIFEST_DIR) if "squam" in f and f.endswith(".txt")]
    adeno_files = [f for f in os.listdir(MANIFEST_DIR) if "adeno" in f and f.endswith(".txt")]

    # ---- Squamous ----
    for f in squam_files:
        process_manifest(f, n_squam, mode, status_choice, OUT_DIR)

    # ---- Adeno ----
    if len(adeno_files) == 0:
        print("⚠️ No adeno manifest files found.")
    else:
        base_n = n_adeno_total // len(adeno_files)
        n_per_adeno = [base_n] * len(adeno_files)
        for i in range(n_adeno_total % len(adeno_files)):
            n_per_adeno[i] += 1

        print("\n📊 Adeno distribution plan:")
        for f, n in zip(adeno_files, n_per_adeno):
            print(f"  {f}: {n} samples")

        for f, n in zip(adeno_files, n_per_adeno):
            process_manifest(f, n, mode, status_choice, OUT_DIR)

    # ---- Organize ----
    organize_by_type(OUT_DIR)

    print("\n✅ Done! Files organized in:")
    print(f"  {os.path.join(OUT_DIR, 'adeno')}")
    print(f"  {os.path.join(OUT_DIR, 'squamous')}")

if __name__ == "__main__":
    main()
