import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Define directories
input_dir = "chest_xray"  # Your dataset directory
output_dir_before = "output/before"
output_dir_after = "output/after"

# Create output directories if they don't exist
os.makedirs(output_dir_before, exist_ok=True)
os.makedirs(output_dir_after, exist_ok=True)

# Define preprocessing pipeline
print("=" * 70)
print("PREPROCESSING PIPELINE CONFIGURATION")
print("=" * 70)
print("\n🔧 TRAINING DATA PIPELINE (with augmentation):")
print("  1. Resize to (224, 224) - Standard input size for ResNet50")
print("  2. Grayscale to RGB - Convert single channel to 3 channels")
print("  3. RandomHorizontalFlip(p=0.5) - 50% chance of horizontal flip")
print("  4. RandomRotation(±15°) - Random rotation for better generalization")
print("  5. RandomAffine - Random translation and scaling")
print("     - Translation: ±10% in both directions")
print("     - Scale: 0.9 to 1.1")
print("  6. ColorJitter - Random brightness and contrast adjustment")
print("     - Brightness: ±20%")
print("     - Contrast: ±20%")
print("  7. ToTensor - Convert PIL Image to PyTorch tensor [0, 1]")
print("  8. Normalize - Apply ImageNet statistics (mean/std)")
print("     - Mean: [0.485, 0.456, 0.406]")
print("     - Std:  [0.229, 0.224, 0.225]")
print("\n🔧 TEST/VAL DATA PIPELINE (no augmentation):")
print("  1. Resize to (224, 224)")
print("  2. Grayscale to RGB")
print("  3. ToTensor")
print("  4. Normalize - Same ImageNet statistics")
print("=" * 70)
print()

# Training preprocessing with augmentation
train_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
    transforms.RandomRotation(15),  # Random rotation up to ±15 degrees
    transforms.RandomAffine(
        degrees=0,  # No additional rotation (already applied above)
        translate=(0.1, 0.1),  # Random translation up to 10%
        scale=(0.9, 1.1)  # Random scaling between 90% and 110%
    ),
    transforms.ColorJitter(
        brightness=0.2,  # Random brightness adjustment ±20%
        contrast=0.2  # Random contrast adjustment ±20%
    ),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Test/Val preprocessing without augmentation
test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Function to unnormalize and convert tensor to PIL image for saving
def unnormalize(tensor):
    """
    Reverse ImageNet normalization to convert tensor back to viewable image.
    
    Process:
    1. Clone tensor to avoid modifying original
    2. Reverse normalization: pixel = (normalized_pixel * std) + mean
    3. Clamp values to [0, 1] range for valid image
    4. Convert tensor to PIL Image for saving
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.clone()  # Avoid modifying original tensor
    tensor.mul_(std).add_(mean)  # Reverse normalization
    tensor = tensor.clamp(0, 1)  # Ensure pixel values are in [0, 1]
    return transforms.ToPILImage()(tensor)

# Function to preprocess and save images
def preprocess_and_save_images(input_dir, output_dir_before, output_dir_after):
    """
    Process all images in the dataset through the preprocessing pipeline.
    
    Steps for each image:
    1. Load original image from disk
    2. Save copy of original to 'before' directory
    3. Apply preprocessing transforms (resize, RGB conversion, augmentation, normalization)
    4. Unnormalize for visualization/storage
    5. Save preprocessed image to 'after' directory
    
    Dataset structure: input_dir/split/class/images
    """
    # Process train, test, and val splits
    splits = ['train', 'test', 'val']
    
    total_processed = 0
    total_errors = 0
    
    for split in splits:
        split_path = os.path.join(input_dir, split)
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split_path} not found, skipping...")
            continue
        
        # Select appropriate preprocessing pipeline
        if split == 'train':
            preprocess = train_preprocess
            pipeline_type = "TRAINING (with augmentation)"
        else:
            preprocess = test_preprocess
            pipeline_type = "TEST/VAL (no augmentation)"
            
        print(f"\n{'='*70}")
        print(f"📂 Processing {split.upper()} split...")
        print(f"🔧 Using {pipeline_type} pipeline")
        print(f"{'='*70}")
        
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Skip macOS hidden files
            if class_dir.startswith('.'):
                print(f"  ⏭️  Skipping hidden directory: {class_dir}")
                continue
                
            print(f"\n  📁 Class: {class_dir}")
            print(f"  {'─'*66}")
            
            # Create split and class subdirectories in output
            before_dir = os.path.join(output_dir_before, split, class_dir)
            after_dir = os.path.join(output_dir_after, split, class_dir)
            os.makedirs(before_dir, exist_ok=True)
            os.makedirs(after_dir, exist_ok=True)
            print(f"  ✓ Created output directories:")
            print(f"    - Before: {before_dir}")
            print(f"    - After:  {after_dir}")
            
            # Count total images first
            all_files = os.listdir(class_path)
            image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
            total_images = len(image_files)
            print(f"  📊 Found {total_images} images to process")
            print()
            
            img_count = 0
            error_count = 0
            
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                try:
                    # Step 1: Load original image
                    original_img = Image.open(img_path)
                    original_size = original_img.size
                    original_mode = original_img.mode
                    
                    # Step 2: Save original image
                    before_path = os.path.join(before_dir, img_name)
                    original_img.save(before_path)
                    
                    # Step 3: Apply preprocessing pipeline
                    preprocessed_tensor = preprocess(original_img)
                    
                    # Step 4: Unnormalize for visualization
                    preprocessed_img = unnormalize(preprocessed_tensor)
                    
                    # Step 5: Save preprocessed image
                    after_path = os.path.join(after_dir, img_name)
                    preprocessed_img.save(after_path)
                    
                    img_count += 1
                    
                    # Verbose progress update
                    if img_count == 1:
                        print(f"  🔄 Processing images...")
                        print(f"     First image: {img_name}")
                        print(f"       • Original size: {original_size}, mode: {original_mode}")
                        print(f"       • Preprocessed size: (224, 224), mode: RGB")
                        if split == 'train':
                            print(f"       • Transforms applied: Resize → Grayscale→RGB → HFlip → Rotation(±15°)")
                            print(f"         → Affine(translate/scale) → ColorJitter → ToTensor → Normalize")
                        else:
                            print(f"       • Transforms applied: Resize → Grayscale→RGB → ToTensor → Normalize (no augmentation)")
                    
                    if img_count % 100 == 0:
                        progress = (img_count / total_images) * 100
                        print(f"     ⏳ Progress: {img_count}/{total_images} ({progress:.1f}%)")
                        
                except Exception as e:
                    error_count += 1
                    print(f"     ❌ Error processing {img_name}: {e}")
            
            # Summary for this class
            print(f"\n  ✅ Completed {split}/{class_dir}:")
            print(f"     • Successfully processed: {img_count} images")
            if error_count > 0:
                print(f"     • Errors encountered: {error_count} images")
            print(f"     • Success rate: {(img_count/(img_count+error_count)*100):.1f}%")
            
            total_processed += img_count
            total_errors += error_count
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total images successfully processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Overall success rate: {(total_processed/(total_processed+total_errors)*100):.1f}%")
    print(f"{'='*70}")

# Function to visualize before and after images
def visualize_before_after(before_path, after_path, title="Before vs After Preprocessing"):
    """
    Create side-by-side visualization of original and preprocessed images.
    
    Args:
        before_path: Path to original image
        after_path: Path to preprocessed image
        title: Title for the figure
    """
    print(f"\n{'='*70}")
    print(f"🎨 VISUALIZATION")
    print(f"{'='*70}")
    
    # Load images
    before_img = Image.open(before_path)
    after_img = Image.open(after_path)
    
    print(f"Original Image:")
    print(f"  • Size: {before_img.size}")
    print(f"  • Mode: {before_img.mode}")
    print(f"  • Path: {before_path}")
    print()
    print(f"Preprocessed Image:")
    print(f"  • Size: {after_img.size}")
    print(f"  • Mode: {after_img.mode}")
    print(f"  • Path: {after_path}")
    print()
    print(f"Changes Applied:")
    print(f"  ✓ Resized from {before_img.size} to {after_img.size}")
    print(f"  ✓ Converted from {before_img.mode} to {after_img.mode}")
    print(f"  ✓ Random rotation applied (±10°)")
    print(f"  ✓ Normalized with ImageNet statistics")
    print(f"  ✓ Unnormalized for visualization")
    print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Before image
    ax1.imshow(before_img, cmap='gray')
    ax1.set_title(f"Before Preprocessing\n{before_img.size} | {before_img.mode}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # After image
    ax2.imshow(after_img)
    ax2.set_title(f"After Preprocessing\n{after_img.size} | {after_img.mode}", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print(f"📊 Displaying visualization...")
    plt.show()
    print(f"{'='*70}")

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🏥 CHEST X-RAY IMAGE PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"📅 Started at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📂 Input Directory: {input_dir}")
    print(f"📂 Output Directories:")
    print(f"   • Original images: {output_dir_before}")
    print(f"   • Preprocessed images: {output_dir_after}")
    print()
    print(f"🎯 Purpose:")
    print(f"   • Prepare chest X-ray images for ResNet50 training")
    print(f"   • Apply standardized preprocessing transforms")
    print(f"   • Save both original and preprocessed versions")
    print()
    print(f"📊 Dataset Structure:")
    print(f"   • Splits: train, val, test")
    print(f"   • Classes: NORMAL, PNEUMONIA")
    print(f"   • Target size: 224x224 (ResNet50 input)")
    print("=" * 70)
    
    # Create output directories
    print(f"\n📁 Creating output directory structure...")
    os.makedirs(output_dir_before, exist_ok=True)
    os.makedirs(output_dir_after, exist_ok=True)
    print(f"   ✓ Directories created successfully")
    
    # Preprocess and save images
    print(f"\n🚀 Starting preprocessing pipeline...\n")
    preprocess_and_save_images(input_dir, output_dir_before, output_dir_after)
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📅 Completed at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📂 Results saved to:")
    print(f"   • {output_dir_before}")
    print(f"   • {output_dir_after}")
    print("=" * 70)
    
    # Visualize a sample image (example: first image from 'NORMAL' class in train split)
    try:
        print(f"\n🔍 Searching for sample image to visualize...")
        sample_split = "train"
        sample_class = "NORMAL"  # Chest X-ray classes are NORMAL and PNEUMONIA
        sample_dir = os.path.join(input_dir, sample_split, sample_class)
        
        print(f"   • Looking in: {sample_split}/{sample_class}")
        
        # Get first valid image file
        sample_img = None
        for f in os.listdir(sample_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
                sample_img = f
                break
        
        if sample_img:
            print(f"   ✓ Found sample: {sample_img}")
            before_path = os.path.join(output_dir_before, sample_split, sample_class, sample_img)
            after_path = os.path.join(output_dir_after, sample_split, sample_class, sample_img)
            visualize_before_after(before_path, after_path, title=f"Sample Visualization: {sample_split}/{sample_class}/{sample_img}")
        else:
            print(f"   ⚠️  No sample image found for visualization.")
    except Exception as e:
        print(f"\n❌ Could not visualize sample: {e}")
    
    print(f"\n{'='*70}")
    print(f"🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")