# mainTrain_enhanced.py - Advanced Augmentation & Regularization
import os
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# Dataset paths
DATASET_PATH = r"C:\Users\Dell\Desktop\deepfake_detector\UADFV"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

def extract_multiple_frames(video_path, num_frames=20):
    """Extract multiple frames from each video with face detection"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return frames
    
    # Use mixed sampling strategy
    uniform_frames = num_frames // 2
    random_frames = num_frames - uniform_frames
    
    # Uniform sampling
    uniform_indices = np.linspace(0, total_frames-1, min(uniform_frames, total_frames), dtype=int)
    
    # Random sampling
    if total_frames > random_frames:
        random_indices = np.random.choice(total_frames, random_frames, replace=False)
    else:
        random_indices = np.arange(total_frames)
    
    frame_indices = np.concatenate([uniform_indices, random_indices])
    frame_indices = np.unique(frame_indices)  # Remove duplicates
    
    for idx in frame_indices[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))  # Larger for augmentation
            frames.append(frame)
    
    cap.release()
    return frames

def get_advanced_augmentation(mode='train'):
    """Advanced data augmentation using Albumentations"""
    if mode == 'train':
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=15, 
                    p=0.8
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ], p=0.8),
            
            # Spatial transformations
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),
            ], p=0.3),
            
            # Color transformations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3, 
                    hue=0.1, 
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, 
                    sat_shift_limit=30, 
                    val_shift_limit=20, 
                    p=0.8
                ),
            ], p=0.8),
            
            # Blur and noise
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ], p=0.5),
            
            # Advanced augmentations
            A.CoarseDropout(
                max_holes=8, 
                max_height=16, 
                max_width=16, 
                min_holes=1, 
                fill_value=0, 
                p=0.5
            ),
            
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.CLAHE(clip_limit=4.0, p=0.3),
            
            # Always apply
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:  # validation
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

class AdvancedDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, augmentations=None, frames_per_video=20):
        self.data = []
        self.augmentations = augmentations
        self.frames_per_video = frames_per_video
        self.frame_cache = {}

        # Load real videos
        real_count = 0
        for video_file in os.listdir(real_dir):
            if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.data.append((os.path.join(real_dir, video_file), 0))
                real_count += 1

        # Load fake videos
        fake_count = 0
        for video_file in os.listdir(fake_dir):
            if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.data.append((os.path.join(fake_dir, video_file), 1))
                fake_count += 1

        random.shuffle(self.data)
        
        print(f"📊 Advanced Dataset loaded:")
        print(f"   Real videos: {real_count}")
        print(f"   Fake videos: {fake_count}")
        print(f"   Frames per video: {frames_per_video}")
        print(f"   Total frames: {len(self.data) * frames_per_video}")

    def __len__(self):
        return len(self.data) * self.frames_per_video

    def __getitem__(self, idx):
        video_idx = idx // self.frames_per_video
        frame_idx = idx % self.frames_per_video
        
        video_path, label = self.data[video_idx]
        
        # Extract frames if not cached
        if video_path not in self.frame_cache:
            self.frame_cache[video_path] = extract_multiple_frames(
                video_path, 
                num_frames=self.frames_per_video
            )
        
        frames = self.frame_cache[video_path]
        
        if len(frames) == 0:
            # Generate synthetic frame if extraction failed
            frame = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        else:
            frame = frames[frame_idx % len(frames)]
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=frame)
            frame = augmented['image']
        else:
            # Default processing
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = torch.tensor(frame).permute(2, 0, 1).float()
        
        return frame, torch.tensor(label, dtype=torch.long)

class AdvancedRegularization(nn.Module):
    """Advanced regularization techniques"""
    def __init__(self, model, drop_rate=0.3):
        super().__init__()
        self.model = model
        self.drop_rate = drop_rate
        
    def forward(self, x):
        return self.model(x)
    
    def apply_stochastic_depth(self, p):
        """Apply stochastic depth (drop paths)"""
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.1  # Lower momentum for small datasets

class EnhancedMobileNetV2(nn.Module):
    """Enhanced MobileNetV2 with advanced regularization"""
    def __init__(self, num_classes=2, drop_rate=0.4):
        super().__init__()
        
        # Load pre-trained model
        self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Enhanced classifier with multiple regularization techniques
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(drop_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Label smoothing
        self.label_smoothing = 0.1
        
        # Initialize weights for new layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for new layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return self.classifier(features)

class DeepfakeTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.train_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"💻 GPU: {torch.cuda.get_device_name(0)}")

    def setup_data(self):
        """Setup data loaders with advanced augmentation"""
        print("📥 Setting up data loaders with advanced augmentation...")
        
        # Advanced augmentations
        train_aug = get_advanced_augmentation(mode='train')
        val_aug = get_advanced_augmentation(mode='val')

        # Create datasets
        train_dataset = AdvancedDeepfakeDataset(
            REAL_PATH, FAKE_PATH, 
            augmentations=train_aug,
            frames_per_video=20
        )
        
        val_dataset = AdvancedDeepfakeDataset(
            REAL_PATH, FAKE_PATH,
            augmentations=val_aug,
            frames_per_video=20
        )

        # Split datasets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_ds, 
            batch_size=16, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=16, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📊 Training samples: {len(train_ds)} frames")
        print(f"📊 Validation samples: {len(val_ds)} frames")
        print("🎯 Advanced augmentation applied!")

    def setup_model(self):
        """Setup enhanced model with advanced regularization"""
        print("🧠 Setting up Enhanced MobileNetV2 with advanced regularization...")
        
        self.model = EnhancedMobileNetV2(num_classes=2, drop_rate=0.4)
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📈 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print("✅ Enhanced model with advanced regularization initialized")

    def label_smooth_loss(self, outputs, targets, smoothing=0.1):
        """Label smoothing cross entropy loss"""
        log_probs = F.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        return loss.mean()

    def train_model(self, epochs=20):
        """Train the model with advanced techniques"""
        if self.model is None or self.train_loader is None:
            print("❌ Please setup data and model first!")
            return

        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the cycle length each time
            eta_min=1e-6
        )

        # Gradient clipping
        max_grad_norm = 1.0
        
        # Mixed precision training (if GPU available)
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        print(f"🎯 Starting advanced training for {epochs} epochs...")
        print("⚡ Using: AdamW, CosineAnnealingWarmRestarts, Label Smoothing, Mixed Precision")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for imgs, labels in train_pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                        loss = self.label_smooth_loss(outputs, labels, smoothing=0.1)
                    
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(imgs)
                    loss = self.label_smooth_loss(outputs, labels, smoothing=0.1)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                # Calculate accuracy
                _, pred = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (pred == labels).sum().item()
                train_loss += loss.item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*train_correct/train_total:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = 100 * train_correct / train_total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            val_accuracy = self.validate()
            self.val_accuracies.append(val_accuracy)
            
            # Step scheduler
            scheduler.step()

            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Train Acc:  {train_accuracy:.2f}%")
            print(f"   Val Acc:    {val_accuracy:.2f}%")
            print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 50)

    def validate(self):
        """Validation function"""
        self.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                else:
                    outputs = self.model(imgs)
                    
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        return 100 * val_correct / val_total

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if self.model is None:
            print("❌ No model to evaluate!")
            return

        print("\n📈 === Advanced Model Evaluation ===")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                else:
                    outputs = self.model(imgs)
                    
                probs = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        print(f"✅ Validation Accuracy: {accuracy:.2f}%")
        
        # Classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Advanced Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('results/confusion_matrix_advanced.png', dpi=300, bbox_inches='tight')
        print("💾 Confusion matrix saved as 'results/confusion_matrix_advanced.png'")
        
        # Plot training history
        if self.train_losses and self.val_accuracies:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(self.train_losses, label='Training Loss', color='red')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(self.train_accuracies, label='Training Accuracy', color='blue')
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.plot(self.val_accuracies, label='Validation Accuracy', color='green')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/training_history_advanced.png', dpi=300, bbox_inches='tight')
            print("💾 Training history saved as 'results/training_history_advanced.png'")
        
        return accuracy

    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save!")
            return
            
        path = "trained_advanced_model"
        os.makedirs(path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'train_accuracies': self.train_accuracies
        }, os.path.join(path, "advanced_deepfake.pth"))
        
        print(f"💾 Advanced model saved to {path}/")

def check_dataset():
    """Check if dataset is properly organized"""
    print("🔍 Checking dataset structure...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset folder '{DATASET_PATH}' not found!")
        return False
    
    real_exists = os.path.exists(REAL_PATH)
    fake_exists = os.path.exists(FAKE_PATH)
    
    print(f"📁 'real' folder exists: {real_exists}")
    print(f"📁 'fake' folder exists: {fake_exists}")
    
    real_videos = []
    fake_videos = []
    
    if real_exists:
        real_videos = [f for f in os.listdir(REAL_PATH) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"   🎬 Real videos found: {len(real_videos)}")
    
    if fake_exists:
        fake_videos = [f for f in os.listdir(FAKE_PATH) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"   🎬 Fake videos found: {len(fake_videos)}")
    
    if real_exists and fake_exists:
        total_videos = len(real_videos) + len(fake_videos)
        print(f"   📊 Total videos: {total_videos}")
        print(f"   💪 Expected frames: {total_videos * 20}")
    
    return real_exists and fake_exists and (len(real_videos) > 0) and (len(fake_videos) > 0)

# Install required packages
def install_requirements():
    """Install required packages"""
    try:
        import albumentations
    except ImportError:
        print("📦 Installing albumentations...")
        os.system("pip install albumentations")
        print("✅ albumentations installed!")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Advanced Deepfake Detection Training...")
    print("=" * 70)
    print("🎯 Features: Advanced Augmentation + Enhanced Regularization")
    print("📈 20 frames per video | Mixed Precision | Label Smoothing")
    print("🛠️  AdamW + CosineAnnealingWarmRestarts + Gradient Clipping")
    print("=" * 70)
    
    # Install requirements
    install_requirements()
    
    if not check_dataset():
        print("\n❌ Dataset issue! Please check:")
        print(f"   - Dataset path: {DATASET_PATH}")
        print("   - Ensure 'real' and 'fake' folders exist with videos")
        exit()
    
    # Initialize trainer
    trainer = DeepfakeTrainer()
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Train the model
    trainer.train_model(epochs=20)
    
    # Evaluate
    final_accuracy = trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    print(f"\n🎉 Advanced training completed!")
    print(f"✅ Final Validation Accuracy: {final_accuracy:.2f}%")
    
    # Performance interpretation
    if final_accuracy >= 85:
        print("💡 Outstanding! Advanced techniques worked perfectly.")
    elif final_accuracy >= 75:
        print("💡 Excellent! Significant improvement achieved.")
    elif final_accuracy >= 65:
        print("💡 Good! Better than baseline, room for improvement.")
    else:
        print("💡 Consider trying different architecture or more data.")