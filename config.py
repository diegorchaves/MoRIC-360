"""Configuration and argument parsing for the training pipeline."""

import argparse


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch Image Compression Training")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch-size")
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate"
    )
    parser.add_argument(
        "--lambda_rate", type=float, default=1e-3, metavar="LR", help="weight"
    )
    parser.add_argument(
        "--lambda_rate_list",
        type=float,
        nargs="+",
        default=[1e-3],
        metavar="LR",
        help="list of lambda weights",
    )

    # Model architecture parameters
    parser.add_argument(
        "--hidden_features", type=int, default=64, help="hidden features"
    )
    parser.add_argument("--hidden_layer", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--sythesis_features", type=int, default=12, help="synthesis features"
    )
    parser.add_argument("--mod_hid_layer", type=int, default=0, help="3x3 mod layer")

    # Upsampling parameters
    parser.add_argument(
        "--local_upsampling_kernel_size", type=int, default=8, help="2, 4 or 8"
    )
    parser.add_argument(
        "--upsampling_kernel_size", type=int, default=8, help="2, 4 or 8"
    )
    parser.add_argument(
        "--static_upsampling_kernel",
        default=False,
        help="Use this flag to not learn the upsampling kernel",
    )
    parser.add_argument(
        "--scale", type=int, default=1, help="Predict every scale*1 pixel"
    )

    # Quantization and latent parameters
    parser.add_argument(
        "--latent_factor",
        type=int,
        default=1,
        help="Full resolution -> 1, other W,H/factor",
    )
    parser.add_argument("--mod_base", type=int, default=7, help="Number of base")
    parser.add_argument("--context_arm", type=int, default=16, help="8,16,24,32")
    parser.add_argument("--dim_arm_mod", type=int, default=16, help="arm dimension")

    # Network configuration
    parser.add_argument(
        "--highest_flag",
        type=int,
        default=1,
        help="Full resolution -> 1, other W,H/factor",
    )
    parser.add_argument("--use_candidate", default=True, help="Use candidate")
    parser.add_argument("--sparsity", type=float, default=0.0, help="prune rate")

    # Dataset and paths
    parser.add_argument("--type", default="kodak", help="Dataset type: kodak or clic")
    parser.add_argument(
        "--data", type=str, default="../data", help="Location to store data"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting image index"
    )

    # Mask configuration
    parser.add_argument(
        "--mask_type",
        type=str,
        default="full",
        choices=["full", "lossy", "lossless", "erp"],
        help="Type of mask to use: full (all ones), lossy (lossy regions), lossless (lossless regions), ERP (25, 50, 25)",
    )

    # SWHDC dilations
    parser.add_argument(
        "--swhdc_dilations",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )

    args = parser.parse_args()

    # Set defaults based on dataset type
    if args.type == "kodak":
        args.num_images = 24
    elif args.type == "clic":
        args.num_images = 41
    else:
        raise ValueError(f"Unknown dataset type: {args.type}")

    return args
