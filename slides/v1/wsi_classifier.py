import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import pandas as pd
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"       # Disable all Qt GUI
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false" # Silence Qt warnings
os.environ["DISPLAY"] = ""                        # Prevent X11 usage
import matplotlib
matplotlib.use("Agg")  # Headless backend (no GUI windows)


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
# 1. TISSUE DETECTION AND PATCH EXTRACTION
# ============================================================================

class WSIPatchExtractor:
    """Extract patches from whole slide images with tissue detection"""
    
    def __init__(self, patch_size=224, overlap=0, tissue_threshold=0.7):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.tissue_threshold = tissue_threshold
    
    def detect_tissue_otsu(self, thumbnail, blur_size=7):
        """Use Otsu's method to detect tissue regions"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert (tissue should be white)
        binary = cv2.bitwise_not(binary)
        
        return binary
    
    def is_tissue_patch(self, patch_img, tissue_mask, x, y, level, slide):
        """Check if patch contains enough tissue"""
        # Get corresponding region in tissue mask
        downsample = slide.level_downsamples[level]
        mask_x = int(x / downsample)
        mask_y = int(y / downsample)
        mask_size = int(self.patch_size / downsample)
        
        if mask_x + mask_size > tissue_mask.shape[1] or mask_y + mask_size > tissue_mask.shape[0]:
            return False
        
        patch_mask = tissue_mask[mask_y:mask_y+mask_size, mask_x:mask_x+mask_size]
        tissue_ratio = np.sum(patch_mask > 0) / (mask_size * mask_size)
        
        return tissue_ratio >= self.tissue_threshold
    
    def extract_patches_streaming(self, slide_path, level=0, max_patches=None):
        """
        Extract patches on-the-fly without saving to disk
        Yields: (patch_image, coordinates)
        """
        slide = openslide.OpenSlide(slide_path)
        
        # Get thumbnail for tissue detection
        thumbnail = slide.get_thumbnail((slide.dimensions[0]//32, slide.dimensions[1]//32))
        tissue_mask = self.detect_tissue_otsu(thumbnail)
        
        # Get dimensions at specified level
        level_dims = slide.level_dimensions[level]
        
        patches_extracted = 0
        coords_list = []
        
        # Extract patches
        for y in range(0, level_dims[1] - self.patch_size, self.stride):
            for x in range(0, level_dims[0] - self.patch_size, self.stride):
                # Read patch at specified level
                patch = slide.read_region(
                    (x * int(slide.level_downsamples[level]), 
                     y * int(slide.level_downsamples[level])),
                    level,
                    (self.patch_size, self.patch_size)
                ).convert('RGB')
                
                # Check if patch contains tissue
                if self.is_tissue_patch(patch, tissue_mask, x, y, level, slide):
                    coords_list.append((x, y))
                    yield np.array(patch), (x, y, level)
                    patches_extracted += 1
                    
                    if max_patches and patches_extracted >= max_patches:
                        slide.close()
                        return
        
        slide.close()

# ============================================================================
# 2. CUSTOM DATASET FOR WSI PATCHES
# ============================================================================

class WSIPatchDataset(Dataset):
    """Dataset for WSI patches with multi-resolution support"""
    
    def __init__(self, data_dir, patch_size=224, transform=None, 
                 levels=[0, 1], max_patches_per_slide=100):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.levels = levels
        self.max_patches = max_patches_per_slide
        
        # Find all .svs files
        self.slides = []
        self.slide_labels = []
        
        for class_idx, class_name in enumerate(['adeno', 'squamous']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                svs_files = list(class_dir.glob('*.svs'))
                self.slides.extend(svs_files)
                self.slide_labels.extend([class_idx] * len(svs_files))
        
        print(f"Found {len(self.slides)} slides")
        print(f"  Adeno: {self.slide_labels.count(0)}")
        print(f"  Squamous: {self.slide_labels.count(1)}")
        
        # Pre-extract patches for training
        self.patches = []
        self.patch_coords = []
        self.patch_labels = []
        self.patch_slide_ids = []
        
        self._extract_all_patches()
    
    def _extract_all_patches(self):
        """Extract patches from all slides"""
        extractor = WSIPatchExtractor(patch_size=self.patch_size)
        
        print("\nExtracting patches from slides...")
        for slide_idx, (slide_path, label) in enumerate(zip(self.slides, self.slide_labels)):
            print(f"Processing {slide_path.name} ({slide_idx+1}/{len(self.slides)})")
            
            for level in self.levels:
                patch_count = 0
                for patch_img, coords in extractor.extract_patches_streaming(
                    str(slide_path), level=level, max_patches=self.max_patches):
                    
                    self.patches.append(patch_img)
                    self.patch_coords.append(coords)
                    self.patch_labels.append(label)
                    self.patch_slide_ids.append(slide_idx)
                    patch_count += 1
                
                print(f"  Level {level}: {patch_count} patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = Image.fromarray(self.patches[idx])
        label = self.patch_labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label, self.patch_slide_ids[idx]

# ============================================================================
# 3. DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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
# 4. CREATE DATALOADERS
# ============================================================================

def create_dataloaders(data_dir, batch_size=32, test_split=0.2):
    """Create train/test dataloaders with slide-level split"""
    
    # Create dataset
    full_dataset = WSIPatchDataset(
        data_dir, 
        transform=train_transform,
        levels=[0, 1],
        max_patches_per_slide=50
    )
    
    # Get unique slide IDs
    unique_slides = list(set(full_dataset.patch_slide_ids))
    slide_labels = [full_dataset.slide_labels[s] for s in unique_slides]
    
    # Split at slide level
    train_slides, test_slides = train_test_split(
        unique_slides,
        test_size=test_split,
        random_state=42,
        stratify=slide_labels
    )
    
    # Get patch indices for each split
    train_indices = [i for i, sid in enumerate(full_dataset.patch_slide_ids) 
                    if sid in train_slides]
    test_indices = [i for i, sid in enumerate(full_dataset.patch_slide_ids) 
                   if sid in test_slides]
    
    print(f"\nDataset split:")
    print(f"  Training slides: {len(train_slides)}, patches: {len(train_indices)}")
    print(f"  Testing slides: {len(test_slides)}, patches: {len(test_indices)}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Apply test transform to test dataset
    full_dataset_test = WSIPatchDataset(
        data_dir,
        transform=test_transform,
        levels=[0, 1],
        max_patches_per_slide=50
    )
    test_dataset = Subset(full_dataset_test, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, ['adeno', 'squamous'], test_slides

# ============================================================================
# 5. MODEL ARCHITECTURE
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
# 6. TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, test_loader, class_names, num_epochs=20):
    """Train the model with enhanced visualization"""
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.2f}%'})
        
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
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                'acc': f'{100*correct/total:.2f}%'})
        
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
# 7. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history):
    """Enhanced training history plots"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss plot
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', 
           linewidth=2, markersize=6, color='#3498db')
    ax.plot(epochs, history['test_loss'], 's-', label='Test Loss',
           linewidth=2, markersize=6, color='#e74c3c')
    ax.set_title('Loss Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Accuracy plot
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy',
           linewidth=2, markersize=6, color='#2ecc71')
    ax.plot(epochs, history['test_acc'], 's-', label='Test Accuracy',
           linewidth=2, markersize=6, color='#f39c12')
    ax.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Learning rate plot
    ax = axes[1, 0]
    ax.plot(epochs, history['learning_rate'], 'o-', linewidth=2,
           markersize=6, color='#9b59b6')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Loss difference plot
    ax = axes[1, 1]
    loss_diff = np.array(history['test_loss']) - np.array(history['train_loss'])
    ax.plot(epochs, loss_diff, 'o-', linewidth=2, markersize=6, color='#e67e22')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Overfitting Indicator (Test - Train Loss)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Difference', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history saved as 'training_history.png'")
    plt.close()

def visualize_sample_patches(dataset, num_samples=12):
    """Visualize sample patches from the dataset"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Sample Patches from WSI Dataset', fontsize=16, fontweight='bold')
    
    indices = random.sample(range(len(dataset.patches)), num_samples)
    
    for idx, ax in enumerate(axes.flat):
        patch = dataset.patches[indices[idx]]
        label = dataset.patch_labels[indices[idx]]
        slide_id = dataset.patch_slide_ids[indices[idx]]
        coords = dataset.patch_coords[indices[idx]]
        
        ax.imshow(patch)
        ax.set_title(f"{'Adeno' if label == 0 else 'Squamous'}\n"
                    f"Slide {slide_id}, Lvl {coords[2]}\n"
                    f"Pos: ({coords[0]}, {coords[1]})",
                    fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_patches.png', dpi=300, bbox_inches='tight')
    print("✓ Sample patches saved as 'sample_patches.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Enhanced confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax1, cbar_kws={'label': 'Count'}, square=True)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax2, cbar_kws={'label': 'Percentage'}, square=True)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()

# ============================================================================
# 8. GRAD-CAM VISUALIZATION
# ============================================================================

def visualize_gradcam(model, image_path, class_names, target_layer=None):
    """Generate Grad-CAM visualization"""
    model.eval()
    
    if target_layer is None:
        target_layer = [model.layer4[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layer)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare visualization
    img_np = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Patch', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title(f'Prediction: {class_names[pred_class]} ({probs[pred_class]:.2%})', 
                     fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Grad-CAM visualization saved as 'gradcam_visualization.png'")
    plt.close()

# ============================================================================
# 9. SLIDE-LEVEL PREDICTION
# ============================================================================

def predict_slide(model, slide_path, class_names, level=0, max_patches=200):
    """Predict class for entire slide by aggregating patch predictions"""
    model.eval()
    extractor = WSIPatchExtractor(patch_size=224)
    
    all_probs = []
    patch_coords = []
    
    print(f"\nProcessing slide: {Path(slide_path).name}")
    
    with torch.no_grad():
        for patch_img, coords in tqdm(
            extractor.extract_patches_streaming(slide_path, level=level, max_patches=max_patches),
            desc="Extracting patches"):
            
            # Preprocess patch
            patch_pil = Image.fromarray(patch_img)
            patch_tensor = test_transform(patch_pil).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(patch_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            all_probs.append(probs)
            patch_coords.append(coords)
    
    if len(all_probs) == 0:
        print("Warning: No patches extracted from slide")
        return None
    
    all_probs = np.array(all_probs)
    
    # Aggregate predictions (majority voting and averaging)
    avg_probs = np.mean(all_probs, axis=0)
    pred_class = np.argmax(avg_probs)
    confidence = avg_probs[pred_class]
    
    # Calculate patch-level agreement
    patch_predictions = np.argmax(all_probs, axis=1)
    agreement = np.sum(patch_predictions == pred_class) / len(patch_predictions)
    
    result = {
        'slide_name': Path(slide_path).name,
        'predicted_class': class_names[pred_class],
        'confidence': confidence,
        'patch_agreement': agreement,
        'num_patches': len(all_probs),
        'class_probabilities': {name: prob for name, prob in zip(class_names, avg_probs)},
        'patch_coordinates': patch_coords
    }
    
    print(f"\nSlide Prediction:")
    print(f"  Class: {result['predicted_class']}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Patch Agreement: {agreement:.2%}")
    print(f"  Total Patches: {result['num_patches']}")
    
    return result

def batch_predict_slides(model, data_dir, class_names, output_csv='slide_predictions.csv'):
    """Predict all slides and save results"""
    data_dir = Path(data_dir)
    results = []
    
    print("\n" + "="*80)
    print("BATCH SLIDE PREDICTION".center(80))
    print("="*80 + "\n")
    
    for class_name in ['adeno', 'squamous']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
            
        svs_files = list(class_dir.glob('*.svs'))
        print(f"\nProcessing {len(svs_files)} slides from {class_name}...")
        
        for slide_path in svs_files:
            result = predict_slide(model, str(slide_path), class_names, max_patches=100)
            if result:
                result['true_class'] = class_name
                result['correct'] = result['predicted_class'].lower() == class_name
                results.append(result)
    
    # Save to CSV
    df = pd.DataFrame([{
        'Slide Name': r['slide_name'],
        'True Class': r['true_class'],
        'Predicted Class': r['predicted_class'],
        'Correct': r['correct'],
        'Confidence': f"{r['confidence']:.4f}",
        'Patch Agreement': f"{r['patch_agreement']:.4f}",
        'Num Patches': r['num_patches'],
        'Adeno Prob': f"{r['class_probabilities']['adeno']:.4f}",
        'Squamous Prob': f"{r['class_probabilities']['squamous']:.4f}"
    } for r in results])
    
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")
    
    # Print summary
    if len(results) > 0:
        accuracy = sum(r['correct'] for r in results) / len(results)
        print(f"\nSlide-Level Accuracy: {accuracy:.2%}")
        print(f"Total Slides: {len(results)}")
        print(f"Correct: {sum(r['correct'] for r in results)}")
        print(f"Incorrect: {sum(not r['correct'] for r in results)}")
    
    return results, df

# ============================================================================
# 10. EVALUATION METRICS
# ============================================================================

def evaluate_model(model, test_loader, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n" + "="*80)
    print("MODEL EVALUATION".center(80))
    print("="*80 + "\n")
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Total Test Patches: {len(all_labels)}\n")
    
    print("-"*80)
    print("Classification Report:")
    print("-"*80)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # ROC-AUC if binary classification
    if len(class_names) == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"\nROC-AUC Score: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved as 'roc_curve.png'")
        plt.close()
    
    return all_preds, all_labels, all_probs

# ============================================================================
# 11. MAIN EXECUTION
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    DATA_DIR = 'dataset'
    BATCH_SIZE = 16  # Reduced for memory efficiency with large patches
    NUM_EPOCHS = 20
    
    print("\n" + "="*80)
    print("WSI CANCER CLASSIFICATION SYSTEM".center(80))
    print("Adenocarcinoma vs Squamous Cell Carcinoma".center(80))
    print("="*80 + "\n")
    
    # Step 1: Create dataloaders
    print("[STEP 1/7] Creating dataloaders with patch extraction...")
    print("-"*80)
    train_loader, test_loader, class_names, test_slides = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Step 2: Visualize sample patches
    print("\n[STEP 2/7] Visualizing sample patches...")
    print("-"*80)
    # Access the base dataset from the first loader
    base_dataset = train_loader.dataset.dataset
    visualize_sample_patches(base_dataset)
    
    # Step 3: Create model
    print("\n[STEP 3/7] Creating ResNet50 model...")
    print("-"*80)
    model = create_resnet50_model(num_classes=len(class_names))
    
    # Step 4: Train model
    print("\n[STEP 4/7] Starting training...")
    print("-"*80)
    history = train_model(model, train_loader, test_loader, class_names, num_epochs=NUM_EPOCHS)
    
    # Step 5: Load best model and evaluate
    print("\n[STEP 5/7] Loading best model and evaluating...")
    print("-"*80)
    checkpoint = torch.load('best_wsi_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with test acc {checkpoint['test_acc']:.2f}%\n")
    
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, class_names)
    
    # Step 6: Plot results
    print("\n[STEP 6/7] Generating visualizations...")
    print("-"*80)
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_training_history(history)
    
    # Step 7: Slide-level predictions
    print("\n[STEP 7/7] Generating slide-level predictions...")
    print("-"*80)
    results, df = batch_predict_slides(model, DATA_DIR, class_names)
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!".center(80))
    print("="*80)
    
    print("\nSaved files:")
    print("  - best_wsi_model.pth (trained model checkpoint)")
    print("  - training_history.png (loss and accuracy curves)")
    print("  - confusion_matrix.png (confusion matrix visualization)")
    print("  - roc_curve.png (ROC curve)")
    print("  - sample_patches.png (sample extracted patches)")
    print("  - slide_predictions.csv (slide-level predictions)")
    
    print("\nTo visualize Grad-CAM for a specific patch:")
    print("  visualize_gradcam(model, 'path/to/patch.png', class_names)")
    
    return model, history, results

# ============================================================================
# 12. HELPER FUNCTIONS FOR INFERENCE
# ============================================================================

def quick_slide_predict(model_path, slide_path, class_names=['adeno', 'squamous']):
    """Quick function to predict a single slide after training"""
    model = create_resnet50_model(num_classes=len(class_names), pretrained=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    result = predict_slide(model, slide_path, class_names, max_patches=100)
    return result

def create_prediction_heatmap(model, slide_path, output_path='prediction_heatmap.png', 
                              level=2, max_patches=500):
    """Create a spatial heatmap of predictions across the slide"""
    model.eval()
    extractor = WSIPatchExtractor(patch_size=224)
    
    predictions = []
    coords = []
    
    print(f"Generating prediction heatmap for {Path(slide_path).name}...")
    
    with torch.no_grad():
        for patch_img, patch_coords in tqdm(
            extractor.extract_patches_streaming(slide_path, level=level, max_patches=max_patches),
            desc="Processing patches"):
            
            patch_pil = Image.fromarray(patch_img)
            patch_tensor = test_transform(patch_pil).unsqueeze(0).to(device)
            
            output = model(patch_tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()  # Probability of class 1 (squamous)
            
            predictions.append(prob)
            coords.append(patch_coords[:2])
    
    if len(predictions) == 0:
        print("No patches to visualize")
        return
    
    # Create scatter plot
    coords = np.array(coords)
    predictions = np.array(predictions)
    
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=predictions, 
                         cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Squamous Probability')
    plt.title(f'Prediction Heatmap: {Path(slide_path).name}', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved as '{output_path}'")
    #plt.close()

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Set random seed
    set_seed(42)
    
    # Run main pipeline
    model, history, results = main()
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE".center(80))
    print("="*80)
    
    print("\n1. Quick prediction on a single slide:")
    print("   result = quick_slide_predict('best_wsi_model.pth', 'path/to/slide.svs')")
    
    print("\n2. Generate prediction heatmap:")
    print("   create_prediction_heatmap(model, 'path/to/slide.svs')")
    
    print("\n3. Visualize Grad-CAM for a patch:")
    print("   visualize_gradcam(model, 'path/to/patch.png', ['adeno', 'squamous'])")
    
    print("\n" + "="*80 + "\n")
