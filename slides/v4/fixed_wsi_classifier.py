# enhanced_wsi_classifier_fixed.py
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import pandas as pd
import random
from collections import defaultdict
import warnings
import shutil
import concurrent.futures
warnings.filterwarnings('ignore')
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
os.environ["DISPLAY"] = ""
import matplotlib
matplotlib.use("Agg")

# OpenSlide for .svs files
try:
    import openslide
except ImportError:
    print("OpenSlide not installed. Install with: pip install openslide-python")
    raise

# For Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================================ 
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================ 
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
cv2.setNumThreads(0)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================ 
# 1. IMPROVED TISSUE DETECTION WITH COLOR-BASED FILTERING
# ============================================================================ 

class WSIPatchExtractor:
    """Extract patches from whole slide images with intelligent tissue detection"""
    
    def __init__(self, patch_size=224, overlap=0, tissue_threshold=0.25, white_threshold=0.8):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.tissue_threshold = tissue_threshold  # Minimum tissue percentage
        self.white_threshold = white_threshold    # Maximum white percentage
    
    def detect_tissue_colors(self, patch_img):
        """
        Detect tissue based on H&E staining colors (purple/pink)
        Returns: tissue_percentage, white_percentage, rejection_reason
        """
        patch_np = np.array(patch_img)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(patch_np, cv2.COLOR_RGB2LAB)
        
        # Calculate white percentage (background)
        white_mask = (lab[:, :, 0] > 200)  # Light background in L channel
        white_percentage = np.mean(white_mask)
        
        # Define color ranges for H&E staining
        # Purple/blue range (hematoxylin - nuclei)
        purple_lower1 = np.array([120, 40, 40])   # HSV
        purple_upper1 = np.array([180, 255, 255])
        purple_lower2 = np.array([0, 40, 40])     # Wrap around red-purple
        purple_upper2 = np.array([10, 255, 255])
        
        purple_mask1 = cv2.inRange(hsv, purple_lower1, purple_upper1)
        purple_mask2 = cv2.inRange(hsv, purple_lower2, purple_upper2)
        purple_mask = cv2.bitwise_or(purple_mask1, purple_mask2)
        
        # Pink range (eosin - cytoplasm)
        pink_lower = np.array([140, 30, 30])      # HSV
        pink_upper = np.array([180, 255, 255])
        pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        # Combine tissue masks
        tissue_mask = cv2.bitwise_or(purple_mask, pink_mask)
        tissue_percentage = np.mean(tissue_mask > 0)
        
        # Determine rejection reason
        rejection_reason = None
        if white_percentage > self.white_threshold:
            rejection_reason = f"white_background_{white_percentage:.2f}"
        elif tissue_percentage < self.tissue_threshold:
            rejection_reason = f"low_tissue_{tissue_percentage:.2f}"
        
        return tissue_percentage, white_percentage, rejection_reason
    
    def is_good_patch(self, patch_img):
        """Determine if patch should be kept based on tissue content"""
        tissue_pct, white_pct, rejection_reason = self.detect_tissue_colors(patch_img)
        return rejection_reason is None, tissue_pct, white_pct, rejection_reason
    
    def extract_and_filter_patches(self, slide_path, output_dir, rejected_dir, 
                                 level=0, max_patches_per_slide=None):
        """
        Extract patches and automatically filter them.
        OpenSlide levels are numbered from 0 (highest resolution) to N-1 (lowest resolution).
        - Level 0: Full resolution, highest detail (e.g., 40x magnification).
        - Level 1: Downsampled by a factor (e.g., 4x), making it 10x magnification.
        - And so on. We check if the requested level is valid.
        
        Updated to support random sampling and tissue prioritization:
        - If max_patches_per_slide is set, it shuffles coordinates to sample randomly.
        - It collects a buffer of candidates (2x requested) and selects the ones with highest tissue %.
        """
        try:
            slide = openslide.OpenSlide(slide_path)
        except openslide.OpenSlideError:
            print(f"Could not open slide: {slide_path}. Skipping.")
            return [], [], 0

        slide_name = Path(slide_path).stem
        
        if level >= slide.level_count:
            print(f"Warning: Level {level} not available for slide {slide_name} (max level: {slide.level_count - 1}). Skipping level.")
            return [], [], 0

        # Create output directories
        patch_dir = Path(output_dir) / slide_name / f"level_{level}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        if rejected_dir:
            rejected_patch_dir = Path(rejected_dir) / slide_name / f"level_{level}"
            # rejected_patch_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dimensions at specified level
        level_dims = slide.level_dimensions[level]
        
        patches_info = []
        rejection_log = []
        total_patches_extracted = 0
        
        print(f"Extracting and filtering patches from {slide_name} at level {level}...")
        
        y_coords = range(0, level_dims[1] - self.patch_size, self.stride)
        x_coords = range(0, level_dims[0] - self.patch_size, self.stride)

        # Generate all coordinates first to allow for shuffling
        all_coords = list(itertools.product(x_coords, y_coords))
        
        # Determine sampling strategy
        is_limited_sampling = max_patches_per_slide is not None
        
        if is_limited_sampling:
            # Shuffle to ensure random sampling across the slide
            random.shuffle(all_coords)
            # Collect more than needed to select the best tissue (buffer strategy)
            # We collect 2x the requested amount to have a pool to pick the "best" tissue from
            buffer_multiplier = 2
            collection_goal = max_patches_per_slide * buffer_multiplier
        else:
            collection_goal = float('inf')

        candidates = []

        pbar = tqdm(total=len(all_coords), desc=f"Processing {slide_name} L{level}", leave=False)

        for x, y in all_coords:
            pbar.update(1)
            
            # Check if we have enough candidates in our buffer
            if is_limited_sampling and len(candidates) >= collection_goal:
                break
                    
            # Read patch
            patch = slide.read_region(
                (x * int(slide.level_downsamples[level]), 
                 y * int(slide.level_downsamples[level])),
                level,
                (self.patch_size, self.patch_size)
            ).convert('RGB')
            
            total_patches_extracted += 1
            
            # Check if patch is good
            is_good, tissue_pct, white_pct, rejection_reason = self.is_good_patch(patch)
            
            if is_good:
                # Instead of saving immediately, we add to candidates
                candidate = {
                    'patch': patch,
                    'coordinates': (x, y, level),
                    'slide_name': slide_name,
                    'tissue_percentage': tissue_pct,
                    'white_percentage': white_pct,
                    'status': 'accepted'
                }
                candidates.append(candidate)

            elif rejected_dir:
                # Rejections are logged as before
                base_filename = f"{slide_name}_x{x}_y{y}_l{level}_t{tissue_pct:.2f}_w{white_pct:.2f}.png"
                rejected_filename = f"REJECTED_{rejection_reason}_{base_filename}"
                rejected_path = rejected_patch_dir / rejected_filename
                # patch.save(rejected_path)
                
                rejection_log.append({
                    'patch_path': str(rejected_path),
                    'slide_name': slide_name,
                    'coordinates': (x, y, level),
                    'tissue_percentage': tissue_pct,
                    'white_percentage': white_pct,
                    'rejection_reason': rejection_reason
                })
        
        pbar.close()
        
        # Selection Phase: Prioritize high tissue content
        if is_limited_sampling and candidates:
            # Sort by tissue percentage descending
            candidates.sort(key=lambda c: c['tissue_percentage'], reverse=True)
            # Select top N
            final_selection = candidates[:max_patches_per_slide]
        else:
            final_selection = candidates

        # Saving Phase: Save only the selected patches
        for item in final_selection:
            patch = item.pop('patch') # Remove image object before saving metadata
            tissue_pct = item['tissue_percentage']
            white_pct = item['white_percentage']
            x, y = item['coordinates'][0], item['coordinates'][1]

            # Create informative filename
            base_filename = f"{slide_name}_x{x}_y{y}_l{level}_t{tissue_pct:.2f}_w{white_pct:.2f}.png"
            patch_path = patch_dir / base_filename
            patch.save(patch_path)
            
            item['patch_path'] = str(patch_path)
            patches_info.append(item)

        slide.close()
        
        good_count = len(patches_info)
        rejected_count = total_patches_extracted - good_count # Note: this is approximate as we might stop early
        
        print(f"  Level {level}: {good_count} accepted (selected from {len(candidates)} candidates), {len(rejection_log)} logged rejections")
        
        return patches_info, rejection_log, good_count

# ============================================================================ 
# 2. SIMPLIFIED MANUAL REVIEW TOOLS
# ============================================================================ 

def create_simple_review_report(rejected_dir):
    """Create a simple text report for manual review"""
    
    report_content = f"""
    WSI PATCH REJECTION REPORT
    ==========================
    
    NOTE: Rejected patch images are NOT saved to conserve disk space.
    A detailed log of all rejected patches, including their metadata and
    rejection reasons, is available in the following file:
    
    'rejected_patches_info.csv'
    
    This CSV file contains the following columns:
    - patch_path: The path where the patch would have been saved.
    - slide_name: The name of the source slide.
    - coordinates: The (x, y, level) coordinates of the patch.
    - tissue_percentage: The detected tissue percentage.
    - white_percentage: The detected white background percentage.
    - rejection_reason: The reason why the patch was rejected.
    
    You can use this CSV file for analysis or to identify slides that
    may require parameter adjustments for patch extraction.
    """
    
    # Save report
    report_path = 'manual_review_instructions.txt'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Manual review instructions saved: {report_path}")

    # Since we don't save images, we can't count them here.
    # The count is available when 'rejected_patches_info.csv' is created.
    print(f"✓ Details about rejected patches can be found in 'rejected_patches_info.csv'")
    
    return report_path

def quick_manual_review_sample(rejected_dir, sample_size=50):
    """Quickly review a random sample of rejected patches"""
    rejected_path = Path(rejected_dir)
    
    # Get random sample of rejected patches
    all_rejected = list(rejected_path.rglob('REJECTED_*.png'))
    if not all_rejected:
        print("No rejected patches found!")
        return
    
    sample = random.sample(all_rejected, min(sample_size, len(all_rejected)))
    
    print(f"\nQuick Review of {len(sample)} Random Rejected Patches:")
    print("="*60)
    
    for i, patch_path in enumerate(sample):
        filename = patch_path.name
        # Extract info from filename
        parts = filename.split('_')
        reason = parts[1] if len(parts) > 1 else 'unknown'
        tissue_pct = '?.??'
        white_pct = '?.??'
        
        # Try to extract percentages
        for part in parts:
            if part.startswith('t') and part[1:].replace('.', '').isdigit():
                tissue_pct = part[1:]
            elif part.startswith('w') and part[1:].replace('.', '').isdigit():
                white_pct = part[1:]
        
        print(f"{i+1:2d}. {filename[:50]}...")
        print(f"     Reason: {reason}, Tissue: {tissue_pct}%, White: {white_pct}%")
    
    print("\nIf you see any patches that should be kept, move them manually.")
    print("Otherwise, the automatic filtering is working well!")

# ============================================================================ 
# 3. CUSTOM DATASET FOR SAVED PATCHES
# ============================================================================ 

class SavedPatchDataset(Dataset):
    """Dataset for saved patches with manual filtering capability"""
    
    def __init__(self, patches_dir, transform=None, class_names=['adeno', 'squamous']):
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.class_names = class_names
        
        # Collect all patches
        self.patches = []
        self.labels = []
        self.slide_names = []
        self.patch_paths = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.patches_dir / class_name
            if class_dir.exists():
                # Find all slide directories
                for slide_dir in class_dir.iterdir():
                    if slide_dir.is_dir():
                        # Find all patch images in all levels (skip REJECTED files)
                        for patch_file in slide_dir.rglob('*.png'):
                            if 'REJECTED_' not in patch_file.name:
                                self.patches.append(str(patch_file))
                                self.labels.append(class_idx)
                                self.slide_names.append(slide_dir.name)
                                self.patch_paths.append(str(patch_file))
        
        print(f"Found {len(self.patches)} patches after filtering")
        for class_idx, class_name in enumerate(class_names):
            count = self.labels.count(class_idx)
            print(f"  {class_name}: {count} patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_path = self.patches[idx]
        image = Image.open(patch_path).convert('RGB')
        label = self.labels[idx]
        slide_name = self.slide_names[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, slide_name

# ============================================================================ 
# 4. DATA TRANSFORMS
# ============================================================================ 

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# ============================================================================ 
# 5. CREATE DATALOADERS WITH SLIDE-LEVEL SPLIT
# ============================================================================ 

def create_dataloaders_from_patches(patches_dir, batch_size=32, test_split=0.2):
    """Create train/test dataloaders with proper slide-level split"""
    
    # Create dataset
    full_dataset = SavedPatchDataset(patches_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        raise ValueError("No patches found! Please check if patches were extracted correctly.")
    
    # Get unique slides and their labels
    slide_to_label = {}
    slide_to_indices = defaultdict(list)
    
    for idx, slide_name in enumerate(full_dataset.slide_names):
        label = full_dataset.labels[idx]
        slide_to_label[slide_name] = label
        slide_to_indices[slide_name].append(idx)
    
    unique_slides = list(slide_to_label.keys())
    slide_labels = [slide_to_label[slide] for slide in unique_slides]
    
    print(f"\nFound {len(unique_slides)} unique slides")
    print(f"  Adeno slides: {slide_labels.count(0)}")
    print(f"  Squamous slides: {slide_labels.count(1)}")
    
    # Split at slide level (stratified)
    train_slides, test_slides = train_test_split(
        unique_slides,
        test_size=test_split,
        random_state=42,
        stratify=slide_labels
    )
    
    # Get patch indices for each split
    train_indices = []
    for slide in train_slides:
        train_indices.extend(slide_to_indices[slide])
    
    test_indices = []
    for slide in test_slides:
        test_indices.extend(slide_to_indices[slide])
    
    print(f"\nDataset split:")
    print(f"  Training slides: {len(train_slides)}, patches: {len(train_indices)}")
    print(f"  Testing slides: {len(test_slides)}, patches: {len(test_indices)}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    
    # Create test dataset with test transform
    test_dataset_full = SavedPatchDataset(patches_dir, transform=test_transform)
    test_dataset = Subset(test_dataset_full, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, full_dataset.class_names, test_slides

# ============================================================================ 
# 6. MODEL ARCHITECTURE
# ============================================================================ 

def create_resnet50_model(num_classes=2, pretrained=True):
    """Create ResNet50 with custom classifier"""
    print("Loading pre-trained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Custom classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    print(f"Model created with {num_classes} output classes")
    return model.to(device)

# ============================================================================ 
# 7. TRAINING FUNCTION
# ============================================================================ 

def train_model(model, train_loader, test_loader, class_names, num_epochs=20):
    """Train the model with enhanced visualization and metrics"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=3, verbose=True)
    
    best_test_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rate': []
    }
    
    print("\n" + "="*80)
    print("STARTING TRAINING".center(80))
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{100*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for images, labels, _ in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'class_names': class_names
            }, 'best_wsi_model.pth')
            print(f"✓ Best model saved! (Test Acc: {best_test_acc:.2f}%)\n")
        
        # Update scheduler
        scheduler.step(test_acc)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!".center(80))
    print(f"Best Test Accuracy: {best_test_acc:.2f}%".center(80))
    print("="*80 + "\n")
    
    return history

# ============================================================================ 
# 8. EXTRACT ALL PATCHES FROM SVS FILES WITH AUTOMATIC FILTERING
# ============================================================================ 

def _process_slide_wrapper(args):
    """Helper function for parallel execution to process a single slide."""
    svs_file, data_path, patches_path, rejected_path, is_training_mode, levels, max_patches_per_slide = args
    
    # Each thread needs its own extractor instance
    extractor = WSIPatchExtractor(patch_size=224, tissue_threshold=0.25, white_threshold=0.8)
    
    # Determine class name from parent directory if in training mode
    if is_training_mode and svs_file.parent.name != data_path.name:
        class_name = svs_file.parent.name
    else:
        class_name = 'unknown'
        
    slide_name = svs_file.stem
    
    slide_patches_info = []
    slide_rejection_log = []
    slide_accepted_count = 0
    
    for level in levels:
        output_class_dir = patches_path / class_name if is_training_mode else patches_path
        rejected_class_dir = rejected_path / class_name if rejected_path and is_training_mode else rejected_path

        patches_info, rejection_log, good_count = extractor.extract_and_filter_patches(
            str(svs_file),
            output_class_dir,
            rejected_class_dir,
            level=level,
            max_patches_per_slide=max_patches_per_slide
        )
        slide_patches_info.extend(patches_info)
        slide_rejection_log.extend(rejection_log)
        slide_accepted_count += good_count

    if slide_accepted_count == 0:
        # Return the slide name if no patches were accepted
        return slide_patches_info, slide_rejection_log, slide_name
    
    # Return None for the slide name if patches were found
    return slide_patches_info, slide_rejection_log, None

def extract_all_patches_from_svs(data_dir, patches_dir, rejected_dir, 
                               max_patches_per_slide=200, levels=[0, 1]):
    """Extract patches from all SVS files with automatic filtering using multiple threads."""
    data_path = Path(data_dir)
    patches_path = Path(patches_dir)
    if rejected_dir:
        rejected_path = Path(rejected_dir)
    else:
        rejected_path = None  # Ensure it's None if not provided
    
    all_patches_info = []
    all_rejection_log = []
    slides_with_no_patches = []
    
    print("\n" + "="*80)
    print("EXTRACTING AND FILTERING PATCHES FROM ALL SVS FILES (MULTI-THREADED)".center(80))
    print("="*80 + "\n")
    
    # Check for class subdirectories ('adeno', 'squamous') for training mode
    is_training_mode = any(d.is_dir() and d.name in ['adeno', 'squamous'] for d in data_path.iterdir())
    
    if is_training_mode:
        svs_files = list((data_path / 'adeno').glob('*.svs')) + list((data_path / 'squamous').glob('*.svs'))
    else:
        # Prediction mode: just get all svs files from the input directory
        svs_files = list(data_path.glob('*.svs'))

    total_slides = len(svs_files)
    if total_slides == 0:
        print(f"No .svs files found in {data_path}. Exiting.")
        return [], [], []

    print(f"Found {total_slides} slides to process using up to 8 threads...")

    # Prepare arguments for each worker
    tasks = [(svs_file, data_path, patches_path, rejected_path, is_training_mode, levels, max_patches_per_slide) for svs_file in svs_files]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Use executor.map to process slides in parallel and show progress
        results = list(tqdm(executor.map(_process_slide_wrapper, tasks), total=total_slides, desc="Processing slides"))

    # Aggregate results from all workers
    for patches_info, rejection_log, no_patch_slide_name in results:
        all_patches_info.extend(patches_info)
        all_rejection_log.extend(rejection_log)
        if no_patch_slide_name:
            slides_with_no_patches.append(no_patch_slide_name)

    # Save detailed logs
    if all_patches_info:
        patches_df = pd.DataFrame(all_patches_info)
        patches_df.to_csv('accepted_patches_info.csv', index=False)
        print(f"\n✓ Accepted patches info saved: {len(patches_df)} patches")
    
    if all_rejection_log and rejected_dir:
        rejected_df = pd.DataFrame(all_rejection_log)
        rejected_df.to_csv('rejected_patches_info.csv', index=False)
        print(f"✓ Rejected patches info saved: {len(rejected_df)} patches")
        
        # Print rejection statistics
        rejection_reasons = rejected_df['rejection_reason'].value_counts()
        print("\nRejection Statistics:")
        for reason, count in rejection_reasons.items():
            print(f"  {reason}: {count} patches")

    if slides_with_no_patches:
        print("\nSlides with ZERO accepted patches:")
        for slide_name in slides_with_no_patches:
            print(f"  - {slide_name}")
        with open("slides_with_no_patches.txt", "w") as f:
            for slide_name in slides_with_no_patches:
                f.write(f"{slide_name}\n")
        print("✓ List of slides with no patches saved to 'slides_with_no_patches.txt'")

    total_accepted = len(all_patches_info)
    total_rejected = len(all_rejection_log)
    
    if total_accepted + total_rejected > 0:
        print(f"\n✓ Extraction complete!")
        print(f"  Total accepted: {total_accepted}")
        print(f"  Total rejected: {total_rejected}")
    
    return all_patches_info, all_rejection_log, slides_with_no_patches
# ============================================================================ 
# 9. VISUALIZATION FUNCTIONS
# ============================================================================ 

def visualize_patch_filtering(patches_dir, rejected_dir, num_samples=8):
    """Visualize examples of accepted and rejected patches"""
    
    # Find some accepted patches
    accepted_patches = []
    for patch_file in Path(patches_dir).rglob('*.png'):
        if 'REJECTED_' not in patch_file.name:
            accepted_patches.append(str(patch_file))
    
    # Find some rejected patches
    rejected_patches = []
    if rejected_dir and Path(rejected_dir).exists():
        for patch_file in Path(rejected_dir).rglob('REJECTED_*.png'):
            rejected_patches.append(str(patch_file))
    
    if not accepted_patches and not rejected_patches:
        print("No patches found to visualize.")
        return

    # Sample randomly
    accepted_sample = random.sample(accepted_patches, min(num_samples, len(accepted_patches)))
    rejected_sample = random.sample(rejected_patches, min(num_samples, len(rejected_patches)))
    
    # Create visualization
    num_rows = 2 if rejected_sample else 1
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(20, 4 * num_rows))
    if num_samples == 1 and num_rows > 1:
        axes = axes.reshape(num_rows, 1)
    elif num_samples > 1 and num_rows == 1:
        axes = axes.reshape(1, num_samples)
    elif num_samples == 1 and num_rows == 1:
        axes = np.array([[axes]])


    
    if rejected_sample:
        fig.suptitle('Patch Filtering Results - Accepted vs Rejected', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Accepted Patch Samples', fontsize=16, fontweight='bold')

    # Plot accepted patches
    for i in range(num_samples):
        ax = axes[0, i]
        if i < len(accepted_sample):
            patch_path = accepted_sample[i]
            img = Image.open(patch_path)
            ax.imshow(img)
            ax.set_title(f"Accepted\n{Path(patch_path).name[:30]}...", fontsize=8)
        ax.axis('off')

    # Plot rejected patches only if they exist
    if rejected_sample:
        for i in range(num_samples):
            ax = axes[1, i]
            if i < len(rejected_sample):
                patch_path = rejected_sample[i]
                img = Image.open(patch_path)
                ax.imshow(img)
                # Extract reason from filename
                reason = Path(patch_path).name.split('_')[1] if '_' in Path(patch_path).name else 'unknown'
                ax.set_title(f"Rejected: {reason}\n{Path(patch_path).name[:30]}...", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('patch_filtering_results.png', dpi=300, bbox_inches='tight')
    print("✓ Patch filtering visualization saved as 'patch_filtering_results.png'")
    plt.close()

def plot_training_history(history):
    """Enhanced training history plots with better styling"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create modern style plots
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training History - WSI Classification', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'o-', color='#2E86AB', linewidth=2.5, 
             markersize=6, markerfacecolor='white', markeredgewidth=2, label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 's-', color='#A23B72', linewidth=2.5, 
             markersize=6, markerfacecolor='white', markeredgewidth=2, label='Test Loss')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'o-', color='#18A999', linewidth=2.5, 
             markersize=6, markerfacecolor='white', markeredgewidth=2, label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 's-', color='#F18F01', linewidth=2.5, 
             markersize=6, markerfacecolor='white', markeredgewidth=2, label='Test Accuracy')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    # Plot 3: Learning rate
    ax3.plot(epochs, history['learning_rate'], '^-', color='#6A4C93', linewidth=2.5, 
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#f8f9fa')
    
    # Plot 4: Generalization gap
    gap = np.array(history['test_loss']) - np.array(history['train_loss'])
    ax4.plot(epochs, gap, 'd-', color='#FF6B6B', linewidth=2.5, 
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(epochs, gap, 0, where=(gap >= 0), color='#FF6B6B', alpha=0.3)
    ax4.fill_between(epochs, gap, 0, where=(gap < 0), color='#4ECDC4', alpha=0.3)
    ax4.set_title('Generalization Gap (Test - Train Loss)', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss Difference', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Enhanced training history saved as 'training_history_enhanced.png'")
    plt.close()

def plot_confusion_matrix(results, class_names, output_filename='confusion_matrix.png'):
    """Plots and saves a confusion matrix from slide-level prediction results."""
    true_labels = [res['true_class'] for res in results]
    pred_labels = [res['predicted_class'] for res in results]

    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Slide-Level Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved as '{output_filename}'")
    plt.close()

# ============================================================================
# 10. SLIDE-LEVEL PREDICTION FUNCTIONS
# ============================================================================

def predict_slide_from_patches(model, slide_patches_dir, class_names):
    """Predict class for entire slide by aggregating patch predictions"""
    model.eval()
    
    # Find all patches for this slide
    patch_files = list(Path(slide_patches_dir).rglob('*.png'))
    
    if len(patch_files) == 0:
        print(f"No patches found for slide: {slide_patches_dir}")
        return None
    
    all_probs = []
    patch_predictions = []
    
    print(f"Processing {len(patch_files)} patches from {Path(slide_patches_dir).name}...")
    
    with torch.no_grad():
        for patch_path in tqdm(patch_files, desc="Classifying patches", leave=False):
            try:
                # Load and preprocess patch
                patch_img = Image.open(patch_path).convert('RGB')
                patch_tensor = test_transform(patch_img).unsqueeze(0).to(device)
                
                # Get prediction
                output = model(patch_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                
                all_probs.append(probs)
                patch_pred = np.argmax(probs)
                patch_predictions.append(patch_pred)
                
            except Exception as e:
                print(f"Error processing {patch_path}: {e}")
                continue
    
    if len(all_probs) == 0:
        print("No patches successfully processed")
        return None
    
    all_probs = np.array(all_probs)
    
    # Aggregate predictions
    avg_probs = np.mean(all_probs, axis=0)
    pred_class_idx = np.argmax(avg_probs)
    confidence = avg_probs[pred_class_idx]
    
    # Calculate patch-level agreement
    agreement = np.sum(np.array(patch_predictions) == pred_class_idx) / len(patch_predictions)
    
    result = {
        'slide_name': Path(slide_patches_dir).name,
        'predicted_class': class_names[pred_class_idx],
        'confidence': confidence,
        'patch_agreement': agreement,
        'num_patches': len(all_probs),
        'class_probabilities': {name: float(prob) for name, prob in zip(class_names, avg_probs)},
    }
    
    print(f"\nSlide Prediction Results for {result['slide_name']}:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Patch Agreement: {result['patch_agreement']:.2%}")
    print(f"  Total Patches Analyzed: {result['num_patches']}")
    print(f"  Class Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"    {class_name}: {prob:.4f}")
    
    return result

def batch_predict_all_slides(model, patches_dir, class_names, output_csv='all_slide_predictions.csv'):
    """Predict all slides in the patches directory"""
    patches_path = Path(patches_dir)
    results = []
    
    print("\n" + "="*80)
    print("BATCH PREDICTION FOR ALL SLIDES".center(80))
    print("="*80 + "\n")
    
    # Find all slide directories (works for both training and prediction structure)
    slide_dirs = [d for d in patches_path.glob('*/') if d.is_dir() and any(f.suffix == '.png' for f in d.rglob('*'))]
    if not slide_dirs: # Handle case where patches are in the root of patches_dir (prediction mode)
        slide_dirs = [d for d in patches_path.iterdir() if d.is_dir()]

    for slide_dir in slide_dirs:
        print(f"\nProcessing slide: {slide_dir.name}")
        # Infer true class from path for evaluation during training
        true_class = "unknown"
        if 'adeno' in str(slide_dir).lower():
            true_class = 'adeno'
        elif 'squamous' in str(slide_dir).lower():
            true_class = 'squamous'

        result = predict_slide_from_patches(model, slide_dir, class_names)
        if result:
            result['true_class'] = true_class
            if true_class != 'unknown':
                result['correct'] = result['predicted_class'].lower() == true_class
            results.append(result)
    
    # Save to CSV
    if results:
        df_data = []
        for result in results:
            row = {
                'Slide Name': result['slide_name'],
                'Predicted Class': result['predicted_class'],
                'Confidence': f"{result['confidence']:.4f}",
                'Patch Agreement': f"{result['patch_agreement']:.4f}",
                'Num Patches': result['num_patches'],
            }
            for name, prob in result['class_probabilities'].items():
                row[f'{name.capitalize()} Prob'] = f"{prob:.4f}"
            
            if 'true_class' in result and result['true_class'] != 'unknown':
                row['True Class'] = result['true_class']
                row['Correct'] = result['correct']

            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ All predictions saved to {output_csv}")
        
        # Print summary if we have true labels
        if all('correct' in res for res in results):
            accuracy = sum(result['correct'] for result in results) / len(results)
            print(f"\nOverall Slide-Level Accuracy: {accuracy:.2%}")
            print(f"Total Slides Processed: {len(results)}")
            print(f"Correct Predictions: {sum(result['correct'] for result in results)}")
            print(f"Incorrect Predictions: {sum(not result['correct'] for result in results)}")
        
        return results, df
    else:
        print("No results to save")
        return [], pd.DataFrame()

# ============================================================================ 
# 11. MAIN EXECUTION PIPELINE
# ============================================================================ 

import argparse
import tempfile

def run_training_pipeline(args):
    """Main training and evaluation pipeline"""

    # Configuration
    DATA_DIR = args.data_dir
    PATCHES_DIR = '/mnt/Linux_storage/extracted_patches'
    REJECTED_DIR = 'rejected_patches'
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs

    # OpenSlide levels to extract from. Level 0 is highest resolution.
    LEVELS = [0, 1] 
    print("\n" + "="*80)
    print("AUTOMATED WSI PATCH CLASSIFICATION SYSTEM - TRAINING MODE".center(80))
    print("With Intelligent Patch Filtering".center(80))
    print("="*80 + "\n")

    # Step 1: Extract and filter patches from SVS files
    print("[STEP 1/10] Extracting and filtering patches from SVS files...")
    print("-"*80)
    _, _, _ = extract_all_patches_from_svs(
        DATA_DIR, PATCHES_DIR, REJECTED_DIR, 
        max_patches_per_slide=None, levels=LEVELS
    )

    # Step 2: Visualize filtering results
    print("\n[STEP 2/10] Visualizing patch filtering results...")
    print("-"*80)
    visualize_patch_filtering(PATCHES_DIR, REJECTED_DIR)

    # Step 3: Provide manual review options
    print("\n[STEP 3/10] Generating patch rejection report...")
    print("-"*80)
    create_simple_review_report(REJECTED_DIR)

    # The REJECTED_DIR is no longer created, so no need to remove it.
    # Step 4: Create dataloaders
    print("\n[STEP 4/10] Creating dataloaders...")
    print("-"*80)
    try:
        train_loader, test_loader, class_names, test_slides = create_dataloaders_from_patches(
            PATCHES_DIR,
            batch_size=BATCH_SIZE
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Cannot proceed with training. Please ensure patches were extracted.")
        return

    # Step 5: Create model
    print("\n[STEP 5/10] Creating ResNet50 model...")
    print("-"*80)
    model = create_resnet50_model(num_classes=len(class_names))

    # Step 6: Train model
    print("\n[STEP 6/10] Starting training...")
    print("-"*80)
    history = train_model(model, train_loader, test_loader, class_names, num_epochs=NUM_EPOCHS)

    # Step 7: Load best model
    print("\n[STEP 7/10] Loading best model...")
    print("-"*80)
    checkpoint = torch.load('best_wsi_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with test acc {checkpoint['test_acc']:.2f}%\n")

    # Step 8: Generate predictions on the test set
    print("\n[STEP 8/10] Generating slide-level predictions for the test set...")
    print("-"*80)
    results = []

    # Find the true class for each test slide
    slide_to_class = {}
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(PATCHES_DIR) / class_name
        if class_dir.exists():
            for slide_dir in class_dir.iterdir():
                if slide_dir.name in test_slides:
                    slide_to_class[slide_dir.name] = class_name

    for slide_name in tqdm(test_slides, desc="Predicting on test slides"):
        true_class = slide_to_class.get(slide_name)
        if not true_class:
            print(f"Warning: Could not find class for test slide {slide_name}. Skipping.")
            continue
        slide_dir = Path(PATCHES_DIR) / true_class / slide_name
        result = predict_slide_from_patches(model, slide_dir, class_names)
        if result:
            result['true_class'] = true_class
            result['correct'] = result['predicted_class'].lower() == true_class.lower()
            results.append(result)

    # Save to CSV
    if results:
        df_data = []
        for result in results:
            row = {
                'Slide Name': result['slide_name'],
                'True Class': result['true_class'],
                'Predicted Class': result['predicted_class'],
                'Correct': result['correct'],
                'Confidence': f"{result['confidence']:.4f}",
                'Patch Agreement': f"{result['patch_agreement']:.4f}",
                'Num Patches': result['num_patches'],
            }
            for name, prob in result['class_probabilities'].items():
                row[f'{name.capitalize()} Prob'] = f"{prob:.4f}"
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv('test_set_predictions.csv', index=False)
        print(f"\n✓ Test set predictions saved to test_set_predictions.csv")

        # Print summary
        accuracy = sum(res['correct'] for res in results) / len(results)
        print(f"\nOverall Slide-Level Accuracy on Test Set: {accuracy:.2%}")
        print(f"Total Slides Processed: {len(results)}")
        print(f"Correct Predictions: {sum(res['correct'] for res in results)}")
        print(f"Incorrect Predictions: {len(results) - sum(res['correct'] for res in results)}")
    else:
        print("No results to save for the test set.")

    # Step 9: Plot results
    print("\n[STEP 9/10] Generating training visualizations...")
    print("-"*80)
    plot_training_history(history)

    # Step 10: Plot confusion matrix
    print("\n[STEP 10/10] Generating confusion matrix...")
    print("-"*80)
    if results:
        plot_confusion_matrix(results, class_names)
    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETE!".center(80))
    print("="*80)
    print("\nSaved files:")
    print("  - best_wsi_model.pth (trained model)")
    print("  - accepted_patches_info.csv (accepted patches details)")
    print("  - rejected_patches_info.csv (rejected patches details)")
    print("  - patch_filtering_results.png (filtering visualization)")
    print("  - manual_review_instructions.txt (review instructions)")
    print("  - training_history_enhanced.png (training curves)")
    print("  - test_set_predictions.csv (test set slide predictions)")
    print("  - confusion_matrix.png (slide-level confusion matrix)")
    print("  - slides_with_no_patches.txt (slides with no good patches)")

def run_prediction_pipeline(args):
    """Runs prediction on a folder of SVS files."""
    MODEL_PATH = args.model_path
    INPUT_DIR = args.input_dir
    OUTPUT_CSV = args.output_csv
    LEVELS = [0, 1] # Levels to extract patches from
    
    # Determine max_patches_per_slide from args
    if args.num_patches.lower() == 'all':
        max_patches_per_slide = None
        print("Extracting all acceptable patches per slide.")
    else:
        try:
            max_patches_per_slide = int(args.num_patches)
            print(f"Extracting up to {max_patches_per_slide} patches per slide per level.")
        except ValueError:
            print(f"Error: --num-patches must be 'all' or an integer. Got '{args.num_patches}'.")
            return

    print("\n" + "="*80)
    print("AUTOMATED WSI PATCH CLASSIFICATION SYSTEM - PREDICTION MODE".center(80))
    print("="*80 + "\n")

    # Step 1: Load model
    print(f"[STEP 1/3] Loading model from {MODEL_PATH}...")
    print("-"*80)
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get('class_names', ['adeno', 'squamous']) # Default if not in checkpoint
    model = create_resnet50_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded model trained for {len(class_names)} classes: {class_names}\n")

    # Step 2: Extract patches from input SVS files
    patches_dir = Path('/mnt/DATA/extracted_patches_pred')
    print(f"[STEP 2/3] Extracting patches to permanent directory: {patches_dir}")
    print("-"*80)
    
    extract_all_patches_from_svs(
        data_dir=INPUT_DIR,
        patches_dir=patches_dir,
        rejected_dir=None, # Don't save rejected patches in prediction mode
        max_patches_per_slide=max_patches_per_slide,
        levels=LEVELS
    )

    # Step 3: Generate predictions
    print("\n[STEP 3/3] Generating slide-level predictions...")
    print("-"*80)
    if not any(patches_dir.iterdir()):
        print("No patches were extracted. Cannot run prediction.")
    else:
        _, _ = batch_predict_all_slides(model, patches_dir, class_names, output_csv=OUTPUT_CSV)

    print("\n" + "="*80)
    print("✓ PREDICTION PIPELINE COMPLETE!".center(80))
    print("="*80)
    print(f"\nPredictions saved to: {OUTPUT_CSV}")

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Automated WSI Patch Classification System")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode: train or predict')

    # --- Training Mode ---
    parser_train = subparsers.add_parser('train', help='Run the full training pipeline')
    parser_train.add_argument('--data-dir', type=str, default='dataset', help='Directory with adeno/ and squamous/ subfolders containing .svs files')
    parser_train.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser_train.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser_train.set_defaults(func=run_training_pipeline)

    # --- Prediction Mode ---
    parser_predict = subparsers.add_parser('predict', 
        help='Predict on a folder of new .svs files. Parameters: --input-dir (required), --model-path, --output-csv, --num-patches',
        description='Predict mode: Classify .svs slides in a directory. Requires --input-dir.')
    parser_predict.add_argument('--model-path', type=str, default='best_wsi_model.pth', help='Path to the trained model (.pth file)')
    parser_predict.add_argument('--input-dir', type=str, required=True, help='Directory containing .svs files to predict on')
    parser_predict.add_argument('--output-csv', type=str, default='predictions.csv', help='Path to save the output predictions CSV file')
    parser_predict.add_argument('--num-patches', type=str, default='all', help='Number of patches per level ("all" or an integer). Default: "all"')
    parser_predict.set_defaults(func=run_prediction_pipeline)
    args = parser.parse_args()
    args.func(args)