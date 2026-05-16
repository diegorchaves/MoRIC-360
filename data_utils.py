"""Data loading and preprocessing utilities."""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import numpy as np


def get_image_paths(args, image_index):
    """
    Get paths for the image based on dataset type and index.
    
    Args:
        args: configuration object with type and dataset paths
        image_index: index of the image (0-based)
    
    Returns:
        tuple: (val_folder, lossy_path, lossless_path)
    """
    idx_str = f"{image_index + 1:02d}"
    
    if args.type == 'kodak':
        val_folder = f'./dataset/kodak_data_set/kodim{idx_str}'
        lossy_path = f'./dataset/kodak_data_set/kodak_lossy_mask/kodim{idx_str}.png'
        lossless_path = f'./dataset/kodak_data_set/kodak_mask/kodim{idx_str}.png'
    elif args.type == 'clic':
        val_folder = f'./dataset/clic_data_set/clic{idx_str}'
        lossy_path = f'./dataset/clic_data_set/clic_lossy_mask/clic{idx_str}.png'
        lossless_path = f'./dataset/clic_data_set/clic_mask/clic{idx_str}.png'
    else:
        raise ValueError(f"Unknown dataset type: {args.type}")
    
    return val_folder, lossy_path, lossless_path


def load_image(image_folder, batch_size=1):
    """
    Load image from folder.
    
    Args:
        image_folder: path to folder containing the image
        batch_size: batch size for dataloader
    
    Returns:
        tuple: (dataloader, image_tensor, height, width)
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(image_folder, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True
    )
    
    img_in, _ = next(iter(dataloader))
    height, width = img_in.shape[2], img_in.shape[3]
    
    return dataloader, img_in, height, width


def create_mask(height, width, mask_type='full', mask_path=None):
    """
    Create a mask tensor based on the mask type.
    
    Args:
        height: image height
        width: image width
        mask_type: type of mask - 'full' (all ones), 'lossy', or 'lossless'
        mask_path: path to mask image file (required for 'lossy' and 'lossless')
    
    Returns:
        tuple: (target_mask_tensor_2d, target_mask_flat)
    """
    if mask_type == 'full':
        # Create mask where ALL pixels are True (1)
        target_mask = torch.ones((1, 1, height, width), dtype=torch.bool)
        target_mask_flat = target_mask.flatten()
        return target_mask, target_mask_flat
    
    elif mask_type in ['lossy', 'lossless']:
        if mask_path is None:
            raise ValueError(f"mask_path is required for mask_type='{mask_type}'")
        
        # Load mask from image file
        try:
            mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
            mask_array = np.array(mask_img, dtype=np.float32)
            
            # Normalize to [0, 1]
            if mask_array.max() > 0:
                mask_array = mask_array / 255.0
            
            # Convert to boolean (threshold at 0.5)
            mask_bool = mask_array > 0.5
            
            # For 'lossless' type, invert the mask (True = lossless regions)
            if mask_type == 'lossless':
                mask_bool = ~mask_bool
            
            # Reshape to match expected format
            target_mask = torch.from_numpy(mask_bool).unsqueeze(0).unsqueeze(0).bool()
            
            # Ensure correct size
            if target_mask.shape[-2:] != (height, width):
                # Resize if needed
                mask_tensor = target_mask.float()
                mask_resized = torch.nn.functional.interpolate(
                    mask_tensor, size=(height, width), mode='nearest'
                )
                target_mask = (mask_resized > 0.5).bool()
            
            target_mask_flat = target_mask.flatten()
            return target_mask, target_mask_flat
        
        except FileNotFoundError:
            print(f"Warning: Mask file not found at {mask_path}")
            print(f"Falling back to 'full' mask type (all ones)")
            target_mask = torch.ones((1, 1, height, width), dtype=torch.bool)
            target_mask_flat = target_mask.flatten()
            return target_mask, target_mask_flat
    
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


def get_save_path(args, image_index):
    """
    Generate the path for saving the trained model.
    
    Args:
        args: configuration object
        image_index: index of the current image
    
    Returns:
        str: full path to save the model checkpoint
    """
    from common_utils import make_path
    
    folder_path = (f'./saved/modbase_{args.mod_base}/context_{args.context_arm}'
                   f'_arm_mod_{args.dim_arm_mod}')
    make_path(folder_path)
    
    folder_path_ = f'{folder_path}/sparity_{args.sparsity}'
    make_path(folder_path_)
    
    saved_path = (f'{folder_path_}/inr_mod_{args.dim_arm_mod}KODAK_ROI_Global_new_struture_7_'
                  f'{args.sythesis_features}_3_33_orderconcat_operator_{args.mod_hid_layer}'
                  f'_pw_{args.lambda_rate}_img_object{image_index}.pth')
    
    return saved_path
