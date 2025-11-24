import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Subset
import numpy as np
from PIL import Image, ImageFile
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# FIX FOR TRUNCATED IMAGES
# ============================================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 2. LOAD DATASET AND CREATE TRAIN/TEST SPLIT
# ============================================================================

def create_train_test_loaders(data_dir, batch_size=32, test_split=0.2):
    """
    Create train and test loaders from directory structure:
    data_dir/
        dogs/
            img1.jpg
            ...
        wolves/
            img1.jpg
            ...
    """
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir)
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Total images: {len(full_dataset)}")
    
    # Get labels for stratified split
    targets = [label for _, label in full_dataset.samples]
    
    # Create train/test indices with stratification
    train_idx, test_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=test_split,
        random_state=42,
        stratify=targets
    )
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples: {len(test_idx)}")
    
    # Count per class
    train_targets = [targets[i] for i in train_idx]
    test_targets = [targets[i] for i in test_idx]
    
    for i, class_name in enumerate(class_names):
        train_count = train_targets.count(i)
        test_count = test_targets.count(i)
        print(f"  {class_name}: {train_count} train, {test_count} test")
    
    # Create train dataset with augmentation
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_subset = Subset(train_dataset, train_idx)
    
    # Create test dataset without augmentation
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    test_subset = Subset(test_dataset, test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader, class_names

# ============================================================================
# 3. CREATE RESNET50 MODEL
# ============================================================================

def create_resnet50_model(num_classes=2):
    """
    Creates ResNet50 model with transfer learning
    """
    print("Loading pre-trained ResNet50 model...")
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    print(f"Model created with {num_classes} output classes")
    return model.to(device)

# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, test_loader, class_names, num_epochs=15):
    """
    Train the model and evaluate on test set
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_test_accuracy = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # ============= TRAINING PHASE =============
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print('-' * 60)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100 * correct / total
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        
        # ============= TESTING PHASE =============
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        
        print(f'\n{"="*60}')
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_accuracy:.2f}%')
        print(f'{"="*60}')
        
        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_resnet50_dogs_wolves.pth')
            print(f'✓ Best model saved! (Test Accuracy: {best_test_accuracy:.2f}%)')
        
        scheduler.step()
    
    print('\n' + '='*60)
    print(f'TRAINING COMPLETE!')
    print(f'Best Test Accuracy: {best_test_accuracy:.2f}%')
    print('='*60)
    
    return history, all_predictions, all_labels

# ============================================================================
# 5. EVALUATE MODEL WITH DETAILED METRICS
# ============================================================================

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate model and return detailed metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Total Test Samples: {len(all_labels)}")
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))
    
    return all_predictions, all_labels, all_probabilities

# ============================================================================
# 6. PLOT CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

# ============================================================================
# 7. PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """
    Plot training and test accuracy/loss curves
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-s', label='Test Loss', linewidth=2)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['test_acc'], 'r-s', label='Test Accuracy', linewidth=2)
    axes[1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()

# ============================================================================
# 8. PREDICTION WITH PROBABILITIES
# ============================================================================

def predict_with_probabilities(model, image_path, class_names):
    """
    Predict single image and show probability percentages
    Output format: "dogs: 25.34%, wolves: 74.66%"
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to convert to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Prediction for: {os.path.basename(image_path)}")
    print('='*60)
    
    for i, class_name in enumerate(class_names):
        percentage = probabilities[i] * 100
        bar = '█' * int(percentage / 2)
        print(f"{class_name:15s}: {percentage:5.2f}% {bar}")
    
    # Get top prediction
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx] * 100
    
    print('-'*60)
    print(f"PREDICTED: {class_names[predicted_idx].upper()} (Confidence: {confidence:.2f}%)")
    print('='*60)
    
    return probabilities, class_names[predicted_idx]

# ============================================================================
# 9. TEST MULTIPLE IMAGES FROM TEST SET
# ============================================================================

def test_sample_images(model, test_loader, class_names, num_samples=5):
    """
    Test a few random images from test set and show predictions
    """
    model.eval()
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    print(f"\n{'='*60}")
    print(f"TESTING {num_samples} SAMPLE IMAGES FROM TEST SET")
    print('='*60)
    
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    for i in range(num_samples):
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        probs = probabilities[i].cpu().numpy()
        
        print(f"\nSample {i+1}:")
        print(f"  True Label: {true_label}")
        for j, class_name in enumerate(class_names):
            print(f"  {class_name}: {probs[j]*100:.2f}%")
        print(f"  → Predicted: {pred_label} {'✓' if true_label == pred_label else '✗'}")
    
    print('='*60)

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete pipeline
    """
    # Configuration
    DATA_DIR = 'dataset'  # Folder containing dogs/ and wolves/
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    TEST_SPLIT = 0.2
    
    print("\n" + "="*60)
    print("ResNet50 - Dogs vs Wolves Classification".center(60))
    print("="*60)
    
    # Step 1: Load data and create train/test split
    print("\n[STEP 1/6] Loading dataset and creating train/test split...")
    print("-"*60)
    train_loader, test_loader, class_names = create_train_test_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        test_split=TEST_SPLIT
    )
    
    # Step 2: Create model
    print("\n[STEP 2/6] Creating ResNet50 model...")
    print("-"*60)
    model = create_resnet50_model(num_classes=len(class_names))
    
    # Step 3: Train model
    print("\n[STEP 3/6] Starting training...")
    print("-"*60)
    history, _, _ = train_model(model, train_loader, test_loader, class_names, num_epochs=NUM_EPOCHS)
    
    # Step 4: Load best model and evaluate
    print("\n[STEP 4/6] Loading best model and evaluating...")
    print("-"*60)
    model.load_state_dict(torch.load('best_resnet50_dogs_wolves.pth'))
    y_pred, y_true, probabilities = evaluate_model(model, test_loader, class_names)
    
    # Step 5: Plot confusion matrix
    print("\n[STEP 5/6] Generating confusion matrix...")
    print("-"*60)
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Step 6: Plot training history
    print("\n[STEP 6/6] Plotting training history...")
    print("-"*60)
    plot_training_history(history)
    
    # Test some sample images
    test_sample_images(model, test_loader, class_names, num_samples=5)
    
    print("\n" + "="*60)
    print("✅ TRAINING AND TESTING COMPLETE!")
    print("="*60)
    print("\nSaved files:")
    print("  - best_resnet50_dogs_wolves.pth (trained model)")
    print("  - training_history.png (accuracy/loss plots)")
    print("  - confusion_matrix.png (confusion matrix heatmap)")
    print("\nTo test on a single image:")
    print("  predict_with_probabilities(model, 'image_path.jpg', class_names)")
    print("="*60)

# ============================================================================
# HELPER FUNCTION: Test a single image after training
# ============================================================================

def test_single_image(model_path, image_path, class_names=['dogs', 'wolves']):
    """
    Quick function to test a single image after training
    
    Usage:
        test_single_image('best_resnet50_dogs_wolves.pth', 'test_image.jpg')
    """
    # Load model
    model = create_resnet50_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predict
    predict_with_probabilities(model, image_path, class_names)

# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()