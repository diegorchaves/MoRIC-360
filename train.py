"""Main training script for MoRIC-360 image compression."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

from common_utils import seed_everything
from config import get_args
from data_utils import create_mask, get_image_paths, get_save_path, load_image
from lossy_contour_algorithm import get_border_bits
from models.candidate_train import train_with_candidates
from models.model import Masked_INR
from trainer import Trainer
from utils.eval_model import eval_model


def train_single_image(args, image_index, trainer):
    """
    Train and evaluate a single image.

    Args:
        args: configuration object
        image_index: index of the image to train
        trainer: Trainer instance

    Returns:
        dict: results containing PSNR, rates, and other metrics
    """
    # Get paths for this image
    val_folder, lossy_path, lossless_path = get_image_paths(args, image_index)

    # Load image and create mask
    dataloader, img_in, height, width = load_image(val_folder, args.batch_size)
    args.patch_h = height
    args.patch_w = width
    args.all_pix_num = height * width
    args.eval_pix_num = height * width

    # Create mask with the specified type
    target_mask, target_mask_flat = create_mask(
        height,
        width,
        mask_type=args.mask_type,
        mask_path=lossless_path if args.mask_type != "full" else None,
    )
    target_mask_flat = target_mask_flat.to("cuda")

    print(f"\n{'=' * 60}")
    print(f"Training image {image_index} (from dataset: {args.type})")
    print(f"Image size: {width}x{height}")
    print(f"Mask type: {args.mask_type}")
    print(f"Arguments: {args}")
    print("=" * 60)

    # Create model
    saved_path = get_save_path(args, image_index)

    if args.use_candidate:
        model = train_with_candidates(args, target_mask, target_mask_flat, dataloader)
    else:
        model = Masked_INR(
            args,
            target_mask,
            sparsity=args.sparsity,
            in_features=2,
            out_features=3 * args.scale * args.scale,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layer,
        )

    model.cuda()
    print(model)
    print(f"Training the {image_index}-th image")

    # Initialize trainer
    trainer.model = model

    # Stage 1 training
    total_steps = 10000
    steps_til_summary = 10

    psnr_s1, checkpoint_s1 = trainer.train_stage_1(
        target_mask_flat,
        dataloader,
        total_steps=total_steps,
        steps_til_summary=steps_til_summary,
    )

    # Stage 2 training
    total_steps_2 = 2000
    psnr_s2, checkpoint_s2 = trainer.train_stage_2(
        target_mask_flat,
        dataloader,
        checkpoint_s1,
        total_steps=total_steps_2,
        steps_til_summary=steps_til_summary,
    )

    # Save checkpoint
    torch.save(checkpoint_s2, saved_path)
    print(f"Saved model at {saved_path}")

    torch.cuda.empty_cache()

    # Evaluation
    psnr_eval, bits_rate, bits_rate_num, _ = trainer.evaluate(
        target_mask_flat, dataloader
    )

    print(
        f"Evaluation - Image {image_index}: PSNR={psnr_eval:.6f}, Rate={bits_rate:.6f}"
    )

    # Load model for extended evaluation
    checkpoint = torch.load(saved_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()

    # Get border bits for lossy contour
    border_rate_num = get_border_bits(lossless_path, image_index)
    border_rate = border_rate_num / args.eval_pix_num

    # Run eval_model for detailed metrics
    (
        eval_psnr,
        eval_y_rate,
        eval_y_rate_num,
        eval_network_rate,
        eval_network_rate_num,
        eval_arm_rate,
        eval_arm_rate_num,
        eval_conv_rate,
        eval_conv_rate_num,
    ) = eval_model(target_mask_flat, args, model, None, dataloader, image_index)

    # Total rates
    total_rate = eval_y_rate + eval_network_rate + border_rate
    total_rate_num = eval_y_rate_num + eval_network_rate_num + border_rate_num

    results = {
        "image_index": image_index,
        "psnr_train": psnr_s1,
        "psnr_eval": eval_psnr,
        "rate_train": bits_rate,
        "rate_train_num": bits_rate_num,
        "y_rate": eval_y_rate,
        "y_rate_num": eval_y_rate_num,
        "network_rate": eval_network_rate,
        "network_rate_num": eval_network_rate_num,
        "arm_rate": eval_arm_rate,
        "arm_rate_num": eval_arm_rate_num,
        "conv_rate": eval_conv_rate,
        "conv_rate_num": eval_conv_rate_num,
        "border_rate": border_rate,
        "border_rate_num": border_rate_num,
        "total_rate": total_rate,
        "total_rate_num": total_rate_num,
    }

    print(f"\nResults for image {image_index}:")
    print(f"  PSNR (eval): {results['psnr_eval']:.6f}")
    print(f"  Total rate: {results['total_rate']:.6f}")
    print(
        f"  Y rate: {results['y_rate']:.6f}, Network rate: {results['network_rate']:.6f}, Border: {results['border_rate']:.6f}"
    )
    print(
        f"  ARM rate: {results['arm_rate']:.6f}, Conv rate: {results['conv_rate']:.6f}"
    )

    return results


def run_training_loop(args):
    """
    Main training loop for all images and lambda rates.

    Args:
        args: configuration object
    """
    # Initialize trainer (will be used for all images)
    trainer = Trainer(None, args, device="cuda")

    # Iterate over lambda rates
    all_results = {}

    for lambda_idx, lambda_rate in enumerate(args.lambda_rate_list):
        print(f"\n\n{'#' * 70}")
        print(
            f"# Lambda rate {lambda_idx + 1}/{len(args.lambda_rate_list)}: {lambda_rate}"
        )
        print(f"{'#' * 70}\n")

        seed_everything(1)
        args.lambda_rate = lambda_rate

        lambda_results = {
            "lambda": lambda_rate,
            "images": [],
            "psnr_list": [],
            "rate_list": [],
            "total_rate_list": [],
        }

        # Train all images with this lambda rate
        for image_idx in range(args.num_images):
            try:
                result = train_single_image(args, image_idx, trainer)
                lambda_results["images"].append(result)
                lambda_results["psnr_list"].append(result["psnr_eval"])
                lambda_results["rate_list"].append(result["y_rate"])
                lambda_results["total_rate_list"].append(result["total_rate"])
            except Exception as e:
                print(f"Error training image {image_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Compute statistics
        if lambda_results["images"]:
            lambda_results["avg_psnr"] = np.mean(lambda_results["psnr_list"])
            lambda_results["avg_rate"] = np.mean(lambda_results["rate_list"])
            lambda_results["avg_total_rate"] = np.mean(
                lambda_results["total_rate_list"]
            )

            print(f"\n\n{'=' * 70}")
            print(f"Summary for lambda = {lambda_rate}:")
            print(f"  Average PSNR: {lambda_results['avg_psnr']:.6f}")
            print(f"  Average latent rate: {lambda_results['avg_rate']:.6f}")
            print(f"  Average total rate: {lambda_results['avg_total_rate']:.6f}")
            print(f"{'=' * 70}\n")

        all_results[lambda_rate] = lambda_results

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for lambda_rate, results in all_results.items():
        if results["images"]:
            print(f"\nLambda = {lambda_rate}:")
            print(f"  Num images: {len(results['images'])}")
            print(f"  PSNR list: {results['psnr_list']}")
            print(f"  Avg PSNR: {results['avg_psnr']:.6f}")
            print(f"  Total rate list: {results['total_rate_list']}")
            print(f"  Avg total rate: {results['avg_total_rate']:.6f}")


def main():
    """Main entry point."""
    args = get_args()
    print(f"Using device: cuda")
    print(f"Loaded configuration: {args}\n")

    run_training_loop(args)


if __name__ == "__main__":
    main()
