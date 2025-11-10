import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.eclairDataloader import EclairDataset
import torch

def test_dataloader():
    print("Testing EclairDataset dataloader...")
    
    # Define the root directory for the dataset
    root_dir = './data/eclair'
    
    # Test with a small sample
    try:
        # Try to create training dataset
        print("Creating training dataset...")
        train_dataset = EclairDataset(
            root_dir=root_dir,
            split='train',
            num_points=4096,
            block_size=20.0,
            sample_rate=0.1,  # Use smaller sample rate for testing
            split_ratio=0.1
        )
        
        print(f"Training dataset created with {len(train_dataset)} samples")
        
        # Test getting a single item
        print("Testing data loading...")
        if len(train_dataset) > 0:
            sample_data, sample_labels = train_dataset[0]
            print(f"Sample data shape: {sample_data.shape}")
            print(f"Sample labels shape: {sample_labels.shape}")
            print("SUCCESS: Training dataloader is working correctly!")
        else:
            print("ERROR: No training samples found")
            
        # Test validation dataset
        print("\nCreating validation dataset...")
        val_dataset = EclairDataset(
            root_dir=root_dir,
            split='val',
            num_points=4096,
            block_size=20.0,
            sample_rate=0.1,
            split_ratio=0.1
        )
        
        print(f"Validation dataset created with {len(val_dataset)} samples")
        
        # Test getting a single item from validation
        if len(val_dataset) > 0:
            sample_data, sample_labels = val_dataset[0]
            print(f"Validation sample data shape: {sample_data.shape}")
            print(f"Validation sample labels shape: {sample_labels.shape}")
            print("SUCCESS: Validation dataloader is working correctly!")
        else:
            print("INFO: No validation samples found (this is normal if split_ratio is small)")
            
    except Exception as e:
        print(f"ERROR: Failed to create or use dataset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()