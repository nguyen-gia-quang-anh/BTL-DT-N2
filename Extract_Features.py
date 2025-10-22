import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import datetime

# Custom Dataset class for loading preprocessed images
class XRayDataset(Dataset):
    """
    Custom Dataset for loading preprocessed X-ray images.
    Images are already preprocessed (224x224, RGB).
    This dataset only applies normalization for ResNet50.
    """
    def __init__(self, root_dir, split='train'):
        """
        Args:
            root_dir: Root directory containing preprocessed images
            split: 'train', 'test', or 'val'
        """
        self.root_dir = root_dir
        self.split = split
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Normalization for ResNet50 (ImageNet stats)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Collect all image paths and labels
        self.samples = []
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"⚠️  Warning: {split_dir} not found")
            return
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')) and not img_name.startswith('.'):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label, img_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, img_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_name

def extract_features_resnet50(preprocessed_dir, batch_size=32, output_dir="features"):
    """
    Extract features from preprocessed images using ResNet50.
    
    Args:
        preprocessed_dir: Directory containing preprocessed images
        batch_size: Batch size for DataLoader
        output_dir: Directory to save extracted features
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🧠 RESNET50 FEATURE EXTRACTION")
    print("=" * 70)
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Load pretrained ResNet50
    print("📥 Loading pretrained ResNet50...")
    resnet50 = models.resnet50(pretrained=True)
    
    # Remove the final classification layer to get features
    # ResNet50 outputs 2048-dimensional features before the FC layer
    feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Set to evaluation mode
    
    print(f"   ✓ ResNet50 loaded successfully")
    print(f"   • Feature dimension: 2048")
    print(f"   • Model parameters: {sum(p.numel() for p in feature_extractor.parameters()):,}")
    print()
    
    # Process each split
    splits = ['train', 'test', 'val']
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"📂 Extracting features from {split.upper()} split...")
        print(f"{'='*70}")
        
        # Create dataset and dataloader
        dataset = XRayDataset(preprocessed_dir, split=split)
        
        if len(dataset) == 0:
            print(f"   ⚠️  No images found in {split} split, skipping...")
            continue
        
        print(f"   📊 Dataset size: {len(dataset)} images")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔢 Number of batches: {len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)}")
        print()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        # Extract features
        all_features = []
        all_labels = []
        all_filenames = []
        
        print(f"   🔄 Extracting features...")
        with torch.no_grad():
            for batch_idx, (images, labels, filenames) in enumerate(tqdm(dataloader, desc=f"   Progress")):
                # Move to device
                images = images.to(device)
                
                # Extract features
                features = feature_extractor(images)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions [B, 2048, 1, 1] -> [B, 2048]
                
                # Store features
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                all_filenames.extend(filenames)
        
        # Concatenate all batches
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        print(f"\n   ✅ Feature extraction complete!")
        print(f"      • Features shape: {all_features.shape}")
        print(f"      • Labels shape: {all_labels.shape}")
        print(f"      • Feature dimension per image: {all_features.shape[1]}")
        print()
        
        # Save features as .npz (compressed)
        npz_path = os.path.join(output_dir, f'{split}_features.npz')
        np.savez_compressed(
            npz_path,
            features=all_features,
            labels=all_labels,
            filenames=all_filenames,
            classes=['NORMAL', 'PNEUMONIA']
        )
        
        print(f"   💾 Saved compressed features to: {npz_path}")
        print(f"      • File size: {os.path.getsize(npz_path) / 1024**2:.2f} MB")
        
        # Also save as separate .npy files for easier access
        npy_dir = os.path.join(output_dir, split)
        os.makedirs(npy_dir, exist_ok=True)
        
        np.save(os.path.join(npy_dir, 'features.npy'), all_features)
        np.save(os.path.join(npy_dir, 'labels.npy'), all_labels)
        np.save(os.path.join(npy_dir, 'filenames.npy'), all_filenames)
        
        print(f"   💾 Also saved as separate .npy files to: {npy_dir}/")
        print(f"      • features.npy: {os.path.getsize(os.path.join(npy_dir, 'features.npy')) / 1024**2:.2f} MB")
        print(f"      • labels.npy: {os.path.getsize(os.path.join(npy_dir, 'labels.npy')) / 1024:.2f} KB")
        print(f"      • filenames.npy: {os.path.getsize(os.path.join(npy_dir, 'filenames.npy')) / 1024:.2f} KB")
        
        # Print statistics
        print(f"\n   📊 Statistics:")
        print(f"      • Total samples: {len(all_labels)}")
        print(f"      • NORMAL: {np.sum(all_labels == 0)} ({np.sum(all_labels == 0)/len(all_labels)*100:.1f}%)")
        print(f"      • PNEUMONIA: {np.sum(all_labels == 1)} ({np.sum(all_labels == 1)/len(all_labels)*100:.1f}%)")
        print(f"      • Feature mean: {all_features.mean():.4f}")
        print(f"      • Feature std: {all_features.std():.4f}")
        print(f"      • Feature min: {all_features.min():.4f}")
        print(f"      • Feature max: {all_features.max():.4f}")
    
    print(f"\n{'='*70}")
    print(f"📊 FEATURE EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ All features extracted and saved to: {output_dir}/")
    print(f"\n✓ Compressed .npz files:")
    for split in splits:
        feature_file = os.path.join(output_dir, f'{split}_features.npz')
        if os.path.exists(feature_file):
            print(f"   • {split}_features.npz ({os.path.getsize(feature_file) / 1024**2:.2f} MB)")
    
    print(f"\n✓ Separate .npy files by split:")
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"   • {split}/")
            print(f"     - features.npy")
            print(f"     - labels.npy")
            print(f"     - filenames.npy")
    
    print()
    print(f"💡 How to load features:")
    print(f"   # Option 1: Load from compressed .npz file")
    print(f"   data = np.load('{output_dir}/train_features.npz')")
    print(f"   features = data['features']  # Shape: (N, 2048)")
    print(f"   labels = data['labels']      # Shape: (N,)")
    print(f"   filenames = data['filenames']")
    print(f"   classes = data['classes']")
    print()
    print(f"   # Option 2: Load from separate .npy files")
    print(f"   features = np.load('{output_dir}/train/features.npy')")
    print(f"   labels = np.load('{output_dir}/train/labels.npy')")
    print(f"   filenames = np.load('{output_dir}/train/filenames.npy')")
    print(f"{'='*70}")

if __name__ == "__main__":
    # Configuration
    preprocessed_dir = "output/after"  # Directory with preprocessed images
    output_dir = "features"             # Directory to save extracted features
    batch_size = 32                     # Batch size for processing
    
    # Extract features
    extract_features_resnet50(
        preprocessed_dir=preprocessed_dir,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"📅 Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
