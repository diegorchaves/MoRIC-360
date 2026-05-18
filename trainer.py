"""Training logic for both stages of the training pipeline."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from common_utils import get_mgrid, loss_to_psnr, ws_mse_loss_flat


class Trainer:
    """Handles two-stage training: Stage 1 with variable temperature/noise, Stage 2 with quantization."""

    def __init__(self, model, args, device="cuda"):
        """
        Initialize the trainer.

        Args:
            model: the neural network model to train
            args: configuration object
            device: device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.args = args
        self.device = device

    def train_stage_1(
        self, target_mask, dataloader, total_steps=100000, steps_til_summary=10
    ):
        """
        Training stage 1: with temperature and noise parameter annealing.

        Args:
            target_mask: binary mask for ROI pixels
            dataloader: data loader for the image
            total_steps: number of training steps
            steps_til_summary: print summary every N steps

        Returns:
            tuple: (best_psnr, checkpoint_dict)
        """
        # Setup optimizer and scheduler
        optim = torch.optim.Adam([p for p in self.model.parameters()], lr=self.args.lr)
        scheduler = CosineAnnealingLR(optim, T_max=total_steps)

        # Get pixel data
        img_in, _ = next(iter(dataloader))
        batch_size, _, height, width = img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).to(self.device)
        pixels1 = pixels[:, target_mask, :]

        # Prepare coordinates
        coords = get_mgrid(width // self.args.scale, height // self.args.scale, 2).to(
            self.device
        )

        # Temperature and noise annealing parameters
        initial_noise_param = 2.0
        final_noise_param = 1.0
        initial_temperature = 0.3
        final_temperature = 0.1

        print(
            "start temperature:",
            initial_temperature,
            "noise parameter:",
            initial_noise_param,
        )
        print(
            "end temperature:", final_temperature, "noise parameter:", final_noise_param
        )
        print("********************Start from stage I")

        best_rd = 1000
        best_psnr = 0
        losses = []
        checkpoint = {}

        for step in range(total_steps + 1):
            self.model.train()

            # Anneal temperature and noise
            self.model.noise_parameter = initial_noise_param - (step / total_steps) * (
                initial_noise_param - final_noise_param
            )
            self.model.soft_round_temperature = initial_temperature - (
                step / total_steps
            ) * (initial_temperature - final_temperature)

            # Forward pass
            model_output, rate, _ = self.model(coords)
            bits_rate = rate.sum() / self.args.all_pix_num
            loss_mse = ws_mse_loss_flat(model_output, pixels, height, width)
            loss = self.args.lambda_rate * bits_rate + loss_mse
            losses.append(loss.item())

            # Log and save best checkpoint
            if not step % steps_til_summary or (step == total_steps - 1):
                psnr_this_iter = loss_to_psnr(loss_mse.item())

                if (loss < best_rd) and (step > 0):
                    best_psnr = psnr_this_iter
                    best_rd = loss
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "binary_mask": None,
                    }
                    print(
                        f"Step {step}, BEST PSNR: {psnr_this_iter:.6f}, Total loss {loss:.6f} "
                        f"with rate {bits_rate.item():.6f}, latent_bits {rate.sum().item():.2f}"
                    )

            # Backward pass
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                10,
                norm_type=2.0,
                error_if_nonfinite=False,
            )
            optim.step()
            scheduler.step()

        return best_psnr, checkpoint

    def train_stage_2(
        self,
        target_mask,
        dataloader,
        checkpoint,
        total_steps=10000,
        steps_til_summary=10,
    ):
        """
        Training stage 2: with fixed quantization (softround alone, no noise).

        Args:
            target_mask: binary mask for ROI pixels
            dataloader: data loader for the image
            checkpoint: checkpoint from stage 1
            total_steps: number of training steps
            steps_til_summary: print summary every N steps

        Returns:
            tuple: (best_psnr_2, checkpoint)
        """
        # Load checkpoint from stage 1
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Setup optimizer and scheduler for stage 2
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=20
        )

        # Get pixel data
        img_in, _ = next(iter(dataloader))
        batch_size, _, height, width = img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).to(self.device)

        # Prepare coordinates
        coords = get_mgrid(width // self.args.scale, height // self.args.scale, 2).to(
            self.device
        )

        print("********************going into stage II")

        best_psnr_2 = 0
        best_rd_2 = 1000
        losses_2 = []

        for step in range(total_steps):
            self.model.train()
            self.model.quantizer_type = "softround_alone"
            self.model.quantizer_noise_type = "none"
            self.model.soft_round_temperature = 1e-4

            # Forward pass
            model_output, rate, _ = self.model(coords)
            bits_rate = rate.sum() / self.args.all_pix_num
            loss_mse = ws_mse_loss_flat(model_output, pixels, height, width)
            loss_2 = self.args.lambda_rate * bits_rate + loss_mse
            losses_2.append(loss_2.item())

            # Log and save best checkpoint
            if not step % steps_til_summary or (step == total_steps - 1):
                psnr_this_iter = loss_to_psnr(loss_mse.item())
                if (loss_2 < best_rd_2) and (step > 0):
                    best_psnr_2 = psnr_this_iter
                    best_rd_2 = loss_2
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "binary_mask": None,
                    }
                    print(f"Rate: {bits_rate}, latent_bits: {rate.sum().item()}")
                    print(
                        f"Step {step}, BEST PSNR: {psnr_this_iter:.6f}, Total loss {loss_2:.6f}"
                    )

            # Backward pass
            optimizer.zero_grad()
            loss_2.backward()
            optimizer.step()
            scheduler.step(loss_2)

            # Early stopping if learning rate drops too low
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < 1e-8:
                print(f"Current learning rate: {current_lr}")
                print(
                    "Stopping training early: Learning rate has dropped below threshold"
                )
                break

        return best_psnr_2, checkpoint

    def evaluate(self, target_mask, dataloader, coords=None):
        """
        Evaluate the model on the image.

        Args:
            target_mask: binary mask for ROI pixels
            dataloader: data loader for the image
            coords: pre-computed coordinates (if None, will be generated)

        Returns:
            tuple: (psnr_eval, bits_rate_eval, bits_rate_eval_num, loss_mse_eval)
        """
        self.model.eval()

        img_in, _ = next(iter(dataloader))
        batch_size, _, height, width = img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).to(self.device)

        if coords is None:
            coords = get_mgrid(
                width // self.args.scale, height // self.args.scale, 2
            ).to(self.device)

        with torch.no_grad():
            model_output, rate, _ = self.model(coords)
            bits_rate_eval = rate.sum() / self.args.all_pix_num

            # loss global — reshape para (B, 3, H, W) internamente
            loss_mse = ws_mse_loss_flat(model_output, pixels, height, width)
            psnr_eval = loss_to_psnr(loss_mse.item())

            # para ROI e background, trabalha no espaço 2D antes de filtrar
            pred_2d = model_output.permute(0, 2, 1).view(-1, 3, height, width)
            target_2d = pixels.permute(0, 2, 1).view(-1, 3, height, width)

            # máscara no formato 2D: (H*W,) → (H, W)
            mask_2d = target_mask.view(height, width)

            pred_roi = pred_2d[:, :, mask_2d]  # (B, 3, N_roi)
            target_roi = target_2d[:, :, mask_2d]

            # aqui já não temos estrutura espacial, então MSE simples é o correto
            # (os pesos por latitude não fazem sentido em pontos sem posição espacial)
            loss_mse_roi = F.mse_loss(pred_roi, target_roi)
            psnr_roi = loss_to_psnr(loss_mse_roi.item())
            print(f"eval_object_psnr: {psnr_roi:.6f}")

            if (~mask_2d).any():
                pred_bg = pred_2d[:, :, ~mask_2d]
                target_bg = target_2d[:, :, ~mask_2d]
                loss_mse_bg = F.mse_loss(pred_bg, target_bg)
                psnr_bg = loss_to_psnr(loss_mse_bg.item())
                print(f"eval_background_psnr: {psnr_bg:.6f}")
            else:
                print("eval_background_psnr: N/A (Entire image is ROI)")

        torch.cuda.empty_cache()

        return (
            psnr_eval,
            bits_rate_eval.item(),
            bits_rate_eval_num.item(),
            loss_mse.item(),
        )
