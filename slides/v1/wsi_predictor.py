import os
import torch
from pathlib import Path
from wsi_classifier import create_resnet50_model, predict_slide
import pandas as pd

# === Handle environments without display ===
# If running headless (no DISPLAY), use non-GUI matplotlib backend
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Safe for OpenCV/Qt
    import matplotlib
    matplotlib.use("Agg")
else:
    import matplotlib
    # Use default GUI backend (TkAgg, Qt5Agg, etc.)
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
MODEL_PATH = "best_wsi_model.pth"
DATA_DIR = "dataset/wwox_deleted"
OUTPUT_CSV = "wwox_deleted_predictions.csv"
CLASS_NAMES = ['adeno', 'squamous']

# === Load trained model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_resnet50_model(num_classes=len(CLASS_NAMES), pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Loaded model from epoch {checkpoint['epoch']+1} (acc={checkpoint['test_acc']:.2f}%)")

# === Predict all .svs files in wwox_deleted ===
data_dir = Path(DATA_DIR)
results = []

for slide_path in data_dir.glob("*.svs"):
    print(f"Processing: {slide_path.name}")
    try:
        result = predict_slide(model, str(slide_path), CLASS_NAMES, max_patches=150)
        if result:
            results.append(result)
    except Exception as e:
        print(f"⚠️ Failed on {slide_path.name}: {e}")

# === Save predictions ===
if not results:
    print("No results found. Check dataset path or model output.")
else:
    df = pd.DataFrame([{
        'Slide Name': r['slide_name'],
        'Predicted Class': r['predicted_class'],
        'Confidence': f"{r['confidence']:.4f}",
        'Patch Agreement': f"{r['patch_agreement']:.4f}",
        'Num Patches': r['num_patches'],
        'Adeno Prob': f"{r['class_probabilities']['adeno']:.4f}",
        'Squamous Prob': f"{r['class_probabilities']['squamous']:.4f}"
    } for r in results])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved predictions to {OUTPUT_CSV}")

    # === Plot distribution ===
    plt.figure(figsize=(6, 4))
    sns.countplot(x='predicted_class', data=df, palette='viridis')
    plt.title("Predicted Class Distribution (wwox_deleted)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("predicted_distribution.png", dpi=200)
    plt.close()
    print("✓ Saved plot to predicted_distribution.png")

print("\n✅ All done!")
