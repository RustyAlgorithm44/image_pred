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
    return pd.read_csv(path, sep="\t")

def subset_manifest(df, n, mode="random"):
    """Returns a subset (random or sequential). If n='all', return entire df."""
    if n == "all":
        return df
    if n > len(df):
        n = len(df)
    return df.sample(n, random_state=42) if mode == "random" else df.head(n)

def write_temp_manifest(df, name):
    temp_path = f"temp_manifest_{name}.txt"
    df.to_csv(temp_path, sep="\t", index=False)
    return temp_path

def gdc_download(manifest_path, outdir):
    cmd = [GDC_CLIENT, "download", "-m", manifest_path, "-d", outdir]
    subprocess.run(cmd, check=True)

def flatten_download_dir(download_dir):
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if f.endswith(".svs"):
                full_path = os.path.join(root, f)
                target_path = os.path.join(download_dir, f)
                if full_path != target_path:
                    shutil.move(full_path, target_path)

    for d in os.listdir(download_dir):
        sub = os.path.join(download_dir, d)
        if os.path.isdir(sub) and not os.listdir(sub):
            os.rmdir(sub)

def rename_files(download_dir, cancer_type):
    for f in os.listdir(download_dir):
        if f.endswith(".svs"):
            parts = f.split(".")
            tcga_id = parts[0].split("-")
            tcga_base = "-".join(tcga_id[:3]) if len(tcga_id) >= 3 else parts[0]
            new_name = f"{tcga_base}_{cancer_type}.svs"
            os.rename(os.path.join(download_dir, f),
                      os.path.join(download_dir, new_name))

def organize_by_type(download_dir):
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
    last_col = df.columns[-1]
    if status_choice == "del":
        return df[df[last_col].str.contains("_del", case=False, na=False)
                  & ~df[last_col].str.contains("not_del", case=False, na=False)]
    elif status_choice == "not_del":
        return df[df[last_col].str.contains("not_del", case=False, na=False)]
    return df

def process_manifest(manifest_file, n_samples, mode, status_choice, out_root):
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

    for f in os.listdir(temp_out):
        if f.endswith(".svs"):
            shutil.move(os.path.join(temp_out, f), os.path.join(out_root, f))
    shutil.rmtree(temp_out)
    os.remove(temp_path)

def parse_num_input(prompt):
    v = input(prompt).strip().lower()
    if v == "all":
        return "all"
    if v == "0" or v == "":
        return 0
    return int(v)

def get_available_manifests():
    """Get all available manifests grouped by type."""
    all_files = [f for f in os.listdir(MANIFEST_DIR) if f.endswith(".txt")]
    
    # First identify squamous files
    squam_files = [f for f in all_files if "squam" in f.lower()]
    
    # Then identify adeno files (excluding squamous)
    adeno_files = [f for f in all_files if f not in squam_files and 
                   ("adeno" in f.lower() or 
                    any(x in f.lower() for x in ["coad", "read", "stad"]))]
    
    return squam_files, adeno_files

def display_manifests_with_counts(manifest_files, status_choice):
    """Display manifests with filtered counts."""
    counts = []
    for f in manifest_files:
        df = read_manifest(os.path.join(MANIFEST_DIR, f))
        df_filtered = filter_by_status(df, status_choice)
        counts.append((f, len(df_filtered)))
    return counts

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GDC DATA DOWNLOAD TOOL - ENHANCED MODE")
    print("=" * 60)
    
    # Get mode and status first
    mode_input = input("\nSubset mode? ('r' = random, 's' = sequential) [default: r]: ").strip().lower()
    mode = "random" if mode_input in ["r", ""] else "sequential"
    
    status_choice = input("Status filter? ('del', 'not_del', or 'all') [default: all]: ").strip().lower()
    if status_choice not in ["del", "not_del", "all"]:
        status_choice = "all"
    
    # Get available manifests
    squam_files, adeno_files = get_available_manifests()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD MODE SELECTION")
    print("=" * 60)
    print("1. Bulk download (specify total for squamous and adeno)")
    print("2. Specific cancer types (target individual cancer types)")
    
    download_mode = input("\nSelect mode (1 or 2) [default: 1]: ").strip()
    
    download_plan = {}  # manifest_file: count
    
    if download_mode == "2":
        # SPECIFIC MODE
        print("\n" + "-" * 60)
        print("SPECIFIC CANCER TYPE SELECTION")
        print("-" * 60)
        
        # Show squamous manifests
        if squam_files:
            print("\nAvailable SQUAMOUS manifests:")
            squam_counts = display_manifests_with_counts(squam_files, status_choice)
            for idx, (f, count) in enumerate(squam_counts, 1):
                print(f"  {idx}. {f}: {count} samples available")
            
            for idx, (f, count) in enumerate(squam_counts, 1):
                n = parse_num_input(f"\n  How many from {f}? (0 to skip, 'all' for all) [default: 0]: ")
                if n != 0:
                    download_plan[f] = n
        
        # Show adeno manifests
        if adeno_files:
            print("\nAvailable ADENO manifests:")
            adeno_counts = display_manifests_with_counts(adeno_files, status_choice)
            for idx, (f, count) in enumerate(adeno_counts, 1):
                print(f"  {idx}. {f}: {count} samples available")
            
            for idx, (f, count) in enumerate(adeno_counts, 1):
                n = parse_num_input(f"\n  How many from {f}? (0 to skip, 'all' for all) [default: 0]: ")
                if n != 0:
                    download_plan[f] = n
    
    else:
        # BULK MODE (original behavior)
        n_squam = parse_num_input("\nEnter number of squamous samples (or 'all') [default: 0]: ")
        n_adeno_total = parse_num_input("Enter number of adeno samples (or 'all') [default: 0]: ")
        
        # Distribute squamous
        for f in squam_files:
            download_plan[f] = n_squam
        
        # Distribute adeno
        if len(adeno_files) > 0 and n_adeno_total != 0:
            if n_adeno_total == "all":
                for f in adeno_files:
                    download_plan[f] = "all"
            else:
                base = n_adeno_total // len(adeno_files)
                remainder = n_adeno_total % len(adeno_files)
                for idx, f in enumerate(adeno_files):
                    download_plan[f] = base + (1 if idx < remainder else 0)
    
    # --------------------------
    # SUMMARY
    # --------------------------
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    print(f"\nSettings:")
    print(f"  Subset mode:    {mode}")
    print(f"  Status filter:  {status_choice}")
    
    print(f"\nPlanned downloads:")
    total_planned = 0
    for manifest_file, count in download_plan.items():
        df = read_manifest(os.path.join(MANIFEST_DIR, manifest_file))
        df_filtered = filter_by_status(df, status_choice)
        available = len(df_filtered)
        
        if count == "all":
            actual = available
            print(f"  {manifest_file}: ALL ({available} samples)")
        else:
            actual = min(count, available)
            print(f"  {manifest_file}: {actual} samples (requested: {count}, available: {available})")
        
        if count != "all":
            total_planned += actual
    
    if any(c == "all" for c in download_plan.values()):
        print(f"\nTotal: Some 'all' selections included")
    else:
        print(f"\nTotal samples to download: {total_planned}")
    
    print("=" * 60)
    
    confirm = input("\nProceed with download? (y/n) [default: n]: ").strip().lower()
    if confirm != "y":
        print("❌ Aborted by user.")
        return
    
    # --------------------------
    # PROCESS MANIFESTS
    # --------------------------
    for manifest_file, count in download_plan.items():
        if count == 0:
            continue
        process_manifest(manifest_file, count, mode, status_choice, OUT_DIR)
    
    # Organize final files
    organize_by_type(OUT_DIR)
    
    print("\n✅ Done! Organized in:")
    print(f"  {os.path.join(OUT_DIR, 'adeno')}")
    print(f"  {os.path.join(OUT_DIR, 'squamous')}")

if __name__ == "__main__":
    main()