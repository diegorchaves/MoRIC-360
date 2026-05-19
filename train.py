import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lossy_contour_algorithm import get_border_bits
from models.candidate_train import train_with_candidates
from models.model import Masked_INR

# ── Logger ──────────────────────────────────────────────────────────────────
from results_logger import ResultsLogger
from utils.eval_model import eval_model, ws_mse, ws_psnr

# ────────────────────────────────────────────────────────────────────────────

manual_seed = 1

loss_mse = None


def seed_everything(seed=1029):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PATHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(1)

print("seed", manual_seed)


def get_mgrid(w_sidelen, h_sidelen, dim=2):
    x = torch.linspace(-1, 1, steps=w_sidelen)
    y = torch.linspace(-1, 1, steps=h_sidelen)
    tensors = (x, y) if dim == 2 else (x,) * dim
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.unsqueeze(0).permute(0, 3, 2, 1)
    return mgrid


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    return 0


def loss_to_psnr(loss, max=1):
    return 10 * np.log10(max**2 / np.asarray(loss))


def get_mask_h_w(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    y_indices, x_indices = np.where(mask == 255)
    target_mask = mask == 255
    if len(x_indices) > 0 and len(y_indices) > 0:
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        cropped_mask = target_mask[min_y : max_y + 1, min_x : max_x + 1]
    return width, height, torch.from_numpy(cropped_mask).unsqueeze(0).unsqueeze(0)


def mm(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    target_mask = mask == 255
    target_mask_flat = target_mask.flatten()
    target_mask_tensor = torch.from_numpy(target_mask_flat).bool()
    return target_mask_tensor, torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0)


def create_mask(height, width, mask_type="full"):
    """
    Create a mask tensor based on the mask type.

    Args:
        height: image height
        width: image width
        mask_type: type of mask
            'full' - all pixels True (no masking effect)
            'erp'  - equatorial 50% True, polar 25% north + 25% south False

    Returns:
        tuple: (target_mask_2d, target_mask_flat)
            target_mask_2d  : BoolTensor shape (1, 1, H, W)
            target_mask_flat: BoolTensor shape (H*W,)
    """
    if mask_type == "full":
        # All pixels are foreground
        target_mask = torch.ones((1, 1, height, width), dtype=torch.bool)
    elif mask_type == "erp":
        # Polar regions (top 25% and bottom 25%) are background (False)
        # Equatorial region (middle 50%) is foreground (True)
        target_mask = torch.zeros(1, 1, height, width, dtype=torch.bool)
        top = int(height * 0.25)
        bottom = int(height * 0.75)
        target_mask[:, :, top:bottom, :] = True
    else:
        raise ValueError(
            f"Unknown mask_type: '{mask_type}'. Choose from 'full' or 'erp'."
        )

    # Flatten to 1D for pixel-indexed indexing used throughout training
    target_mask_flat = target_mask.view(-1)
    return target_mask, target_mask_flat


# ── Dataset genérico para imagens avulsas ────────────────────────────────────
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


class SingleImageDataset(Dataset):
    """Dataset que carrega UMA imagem e a expõe como item único (batch-size 1)."""

    def __init__(self, image_path: str, transform=None):
        self.image_path = image_path
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = Image.open(self.image_path).convert("RGB")
        return self.transform(img), 0  # 0 = dummy label


def load_images_from_dir(images_dir: str):
    """
    Retorna uma lista de (image_name, image_path) para cada imagem
    .png/.jpg/.jpeg encontrada em `images_dir` (não-recursivo).
    """
    base = Path(images_dir)
    if not base.is_dir():
        raise ValueError(f"--images_dir '{images_dir}' não é um diretório válido.")
    entries = sorted(
        p for p in base.iterdir() if p.is_file() and p.suffix in _IMG_EXTENSIONS
    )
    if not entries:
        raise ValueError(
            f"Nenhuma imagem .png/.jpg/.jpeg encontrada em '{images_dir}'."
        )
    return [(p.stem, str(p)) for p in entries]


# ─────────────────────────────────────────────────────────────────────────────


def train(
    target_mask,
    model,
    dataloader,
    total_steps,
    total_steps_2,
    steps_til_summary,
    img_index,
    saved_path,
    logger: ResultsLogger,
    image_name: str,
    args,
):

    criterion = nn.MSELoss().cuda()
    base_params = [p for name, p in model.named_parameters()]

    optim = torch.optim.Adam([{"params": base_params, "lr": args.lr}])
    scheduler = CosineAnnealingLR(optim, T_max=total_steps)

    optimizer_stage_2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    scheduler_stage_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage_2, mode="min", factor=0.8, patience=20
    )

    for batch_idx, (img_in, _) in enumerate(dataloader, 0):
        model.train()

        batch_size, _, height, width = img_in.shape
        pixels = (
            img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).cuda()
        )  # acho que fica (batch_size, height, width, channels) após o permute
        pixels1 = pixels[:, target_mask, :]
        pixels2 = pixels[:, ~target_mask, :]

        coords = get_mgrid(width // args.scale, height // args.scale, 2).cuda()
        losses = []
        losses_2 = []

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
        checkpoint = None
        for step in range(total_steps + 1):
            model.train()
            model.noise_parameter = initial_noise_param - (step / total_steps) * (
                initial_noise_param - final_noise_param
            )
            model.soft_round_temperature = initial_temperature - (
                step / total_steps
            ) * (initial_temperature - final_temperature)

            model_output, rate, _ = model(coords)
            bits_rate = rate.sum() / (args.all_pix_num)
            # loss_mse = criterion(model_output, pixels)
            loss_mse = ws_mse(model_output, pixels, height, width)
            loss = args.lambda_rate * bits_rate + loss_mse
            losses.append(loss.item())

            if not step % steps_til_summary or (step == total_steps - 1):
                # psnr_this_iter = loss_to_psnr(loss_mse.item())
                psnr_this_iter = ws_psnr(model_output, pixels, height, width)
                if (loss < best_rd) and (step > 0):
                    best_rd = loss
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "binary mask": None,
                    }
                    print(
                        "Step %d, BEST PSNR: %0.6f, Total loss %0.6f"
                        % (step, psnr_this_iter, loss),
                        "with its rate",
                        bits_rate.item(),
                        "latent_bits",
                        rate.sum().item(),
                    )

                    # ── Log best checkpoint do stage 1 ──────────────────
                    # logger.log(
                    #     image_name=image_name,
                    #     step="train_stage1_best",
                    #     image_index=img_index,
                    #     lambda_rate=args.lambda_rate,
                    #     args=args,
                    #     metrics={
                    #         "psnr": psnr_this_iter,
                    #         "loss_mse": loss_mse.item(),
                    #         "total_loss": loss.item(),
                    #         "bits_rate": bits_rate.item(),
                    #         "bits_rate_num": rate.sum().item(),
                    #         "train_step": step,
                    #         "noise_param": model.noise_parameter,
                    #         "temperature": model.soft_round_temperature,
                    #     },
                    # )
                    # ────────────────────────────────────────────────────

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                10,
                norm_type=2.0,
                error_if_nonfinite=False,
            )
            optim.step()
            scheduler.step()

        torch.save(checkpoint, saved_path)
        model.load_state_dict(torch.load(saved_path)["model_state_dict"])

        # ── Stage 2 ──────────────────────────────────────────────────────
        print("********************going into stage II")

        best_rd_2 = 1000
        for step in range(total_steps_2):
            model.train()
            model.quantizer_type = "softround_alone"
            model.quantizer_noise_type = "none"
            model.soft_round_temperature = 1e-4

            model_output, rate, _ = model(coords)
            bits_rate = rate.sum() / (args.all_pix_num)
            # loss_mse = criterion(model_output, pixels)
            loss_mse = ws_mse(model_output, pixels, height, width)
            loss_2 = args.lambda_rate * bits_rate + loss_mse
            losses_2.append(loss_2.item())

            if not step % steps_til_summary or (step == total_steps_2 - 1):
                # psnr_this_iter = loss_to_psnr(loss_mse.item())
                psnr_this_iter = ws_psnr(model_output, pixels, height, width)
                if (loss_2 < best_rd_2) and (step > 0):
                    best_rd_2 = loss_2
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "binary mask": None,
                    }
                    print("Print rate", bits_rate)
                    print("latent_bits", rate.sum().item())
                    print(
                        "Step %d, BEST PSNR: %0.6f, Total loss %0.6f"
                        % (step, psnr_this_iter, loss_2)
                    )

                    # ── Log best checkpoint do stage 2 ──────────────────
                    # logger.log(
                    #     image_name=image_name,
                    #     step="train_stage2_best",
                    #     image_index=img_index,
                    #     lambda_rate=args.lambda_rate,
                    #     args=args,
                    #     metrics={
                    #         "psnr": psnr_this_iter,
                    #         "loss_mse": loss_mse.item(),
                    #         "total_loss": loss_2.item(),
                    #         "bits_rate": bits_rate.item(),
                    #         "bits_rate_num": rate.sum().item(),
                    #         "train_step": step,
                    #     },
                    # )
                    # ────────────────────────────────────────────────────

            optimizer_stage_2.zero_grad()
            loss_2.backward()
            optimizer_stage_2.step()
            scheduler_stage_2.step(loss_2)
            current_lr = optimizer_stage_2.param_groups[0]["lr"]

            if current_lr < 1e-8:
                print(f"Current learning rate: {current_lr}")
                print(
                    f"Stopping training early: Learning rate has dropped below lr_threshold"
                )
                break

        torch.cuda.empty_cache()

        # ── Avaliação final ───────────────────────────────────────────────
        model.eval()
        model_output, rate, binary_mask = model(coords)
        bits_rate_eval = rate.sum() / (args.all_pix_num)
        bits_rate_eval_num = rate.sum()
        # loss_mse = criterion(model_output, pixels)
        loss_mse = ws_mse(model_output, pixels, height, width)
        loss_mse_o = criterion(model_output[:, target_mask, :], pixels1)
        # loss_mse_o = ws_mse(model_output[:, target_mask, :], pixels1, height, width)
        loss_mse_b = criterion(model_output[:, ~target_mask, :], pixels2)
        # loss_mse_b = ws_mse(model_output[:, ~target_mask, :], pixels2, height, width)
        psnr_eval = ws_psnr(model_output, pixels, height, width)

        psnr_object = loss_to_psnr(loss_mse_o.item())
        # psnr_object = ws_psnr(model_output[:, target_mask, :], pixels1, height, width)
        # psnr_background = ws_psnr(
        #    model_output[:, ~target_mask, :], pixels2, height, width
        # )
        psnr_background = loss_to_psnr(loss_mse_b.item())

        print("eval_object_psnr:", psnr_object)
        print("eval_background_psnr:", psnr_background)
        print(
            "***** Evaluation Image %d, PSNR: %0.6f, Rate %0.6f *****"
            % (img_index, psnr_eval, bits_rate_eval.item())
        )

        # ── Salva imagem decodificada ─────────────────────────────────────
        # model_output shape: (1, H*W, 3)  →  reshape para (1, H, W, 3)
        decoded_img = model_output.view(1, height, width, 3)
        decoded_path = logger.save_decoded_image(
            image_tensor=decoded_img,
            image_name=image_name,
            lambda_rate=args.lambda_rate,
            step="eval",
            suffix=args.mask_type,
            pos_suffix=args.use_swhdc,
        )
        # ─────────────────────────────────────────────────────────────────

        torch.cuda.empty_cache()
        torch.save(checkpoint, saved_path)
        print("Saved model at", saved_path)

    return psnr_eval, bits_rate_eval.item(), bits_rate_eval_num.item(), decoded_path


# ── Argparse ─────────────────────────────────────────────────────────────────
global args
parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--data", type=str, default="../data")
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--local_upsampling_kernel_size", type=int, default=8)
parser.add_argument("--upsampling_kernel_size", type=int, default=8)
parser.add_argument("--static_upsampling_kernel", default=False)
parser.add_argument("--latent_factor", type=int, default=1)
parser.add_argument("--mod_base", type=int, default=7)
parser.add_argument("--highest_flag", type=int, default=1)
parser.add_argument("--context_arm", type=int, default=16)
parser.add_argument("--dim_arm_mod", type=int, default=16)
parser.add_argument("--use_candidate", default=True)
parser.add_argument("--type", default="kodak")
parser.add_argument("--mod_hid_layer", type=int, default=0)
parser.add_argument("--sythesis_features", type=int, default=12)
parser.add_argument("--hidden_features", type=int, default=64)
parser.add_argument("--hidden_layer", type=int, default=2)
parser.add_argument("--scale", type=int, default=1)
parser.add_argument("--lambda_rate", type=float, default=1e-3)
parser.add_argument("--lambda_rate_list", type=float, nargs="+", default=[1e-3])
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument(
    "--images_dir",
    type=str,
    default=None,
    help=(
        "Caminho para pasta contendo imagens .png/.jpg/.jpeg. "
        "Quando fornecido, ignora --type e carrega todas as imagens da pasta."
    ),
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="full",
    choices=["full", "erp"],
    help=(
        "Tipo de máscara a usar durante o treino. "
        "'full': todos os pixels (sem mascaramento). "
        "'erp': polo norte (25%%) e polo sul (25%%) são background; "
        "equador (50%% central) é foreground."
    ),
)
# ── Novos args do logger ──────────────────────────────────────────────────────
parser.add_argument(
    "--results_dir",
    type=str,
    default="./results",
    help="Diretório raiz para CSV e imagens decodificadas",
)
parser.add_argument(
    "--run_tag", type=str, default="", help="Tag livre para identificar o experimento"
)
# ─────────────────────────────────────────────────────────────────────────────
#
# --- Args SWHDC --------------------------------------------------------------
parser.add_argument("--use_swhdc", action="store_true", default=False)
parser.add_argument("--swhdc_dilations", type=int, nargs="+", default=[1, 2, 3, 4])

args = parser.parse_args()

# --- Monta a lista de imagens a processar ------------------------------------
# Modo genérico: --images_dir aponta para qualquer pasta de imagens
# Modo legado  : --type kodak | clic  (mantido para compatibilidade)
if args.images_dir is not None:
    # Lista de (image_name, image_path) para o modo genérico
    generic_image_list = load_images_from_dir(args.images_dir)
    traing_list = range(len(generic_image_list))
else:
    generic_image_list = None  # sinalizador: usar lógica legada
    if args.type == "kodak":
        traing_list = range(23, 24)
    elif args.type == "clic":
        traing_list = range(0, 41)
    else:
        raise ValueError(
            f"--type '{args.type}' desconhecido. Use 'kodak', 'clic', ou forneça --images_dir."
        )

# ── Instancia o logger UMA vez para todo o experimento ───────────────────────
logger = ResultsLogger(
    base_dir=args.results_dir,
    csv_filename="results.csv",
    run_tag=args.run_tag,
)
# ─────────────────────────────────────────────────────────────────────────────

# (mantém as listas originais para prints finais)
all_psnr_list_of_lists = []
all_rate_list_of_lists = []
all_rate_num_list_of_lists = []
eval_all_psnr_list_of_lists = []
eval_all_y_rate_list_of_lists = []
eval_all_mlp_rate_list_of_lists = []
eval_all_y_rate_num_list_of_lists = []
eval_all_mlp_rate_num_list_of_lists = []
eval_all_border_rate_list_of_lists = []
eval_all_border_rate_num_list_of_lists = []
eval_all_total_rate_list_of_lists = []
eval_all_total_rate_num_list_of_lists = []
eval_all_arm_rate_list_of_lists = []
eval_all_arm_rate_num_list_of_lists = []
eval_all_conv_rate_list_of_lists = []
eval_all_conv_rate_num_list_of_lists = []

for num, lambda_rate in enumerate(args.lambda_rate_list):
    seed_everything(1)
    all_psnr = []
    all_rate = []
    all_rate_num = []
    eval_all_psnr = []
    eval_all_y_rate = []
    eval_all_y_rate_num = []
    eval_all_mlp_rate = []
    eval_all_mlp_rate_num = []
    eval_all_rate_arm = []
    eval_all_rate_arm_num = []
    eval_all_rate_conv = []
    eval_all_rate_conv_num = []
    eval_all_border_rate = []
    eval_all_border_rate_num = []
    eval_all_total_rate = []
    eval_all_total_rate_num = []

    args.lambda_rate = lambda_rate

    for it in traing_list:
        # ── Resolve nome e caminho da imagem ──────────────────────────────
        if generic_image_list is not None:
            # Modo genérico: qualquer pasta de imagens
            image_name, image_path = generic_image_list[it]
            lossyless_path = None  # sem máscara de borda externa
        else:
            # Modo legado: Kodak / CLIC com estrutura de pastas fixa
            idx_str = f"{it + 1:02d}"
            if args.type == "kodak":
                image_name = f"kodim{idx_str}"
                image_path = None  # usa val_folder abaixo
                val_folder = f"./dataset/kodak_data_set/kodim{idx_str}"
                lossyless_path = (
                    f"./dataset/kodak_data_set/kodak_mask/kodim{idx_str}.png"
                )
            elif args.type == "clic":
                image_name = f"clic{idx_str}"
                image_path = None
                val_folder = f"./dataset/clic_data_set/clic{idx_str}"
                lossyless_path = f"./dataset/clic_data_set/clic_mask/clic{idx_str}.png"
        # ─────────────────────────────────────────────────────────────────

        args.lambda_rate = lambda_rate
        transform_val = transforms.Compose([transforms.ToTensor()])

        if image_path is not None:
            # Modo genérico: carrega imagem diretamente do caminho
            val_dataset = SingleImageDataset(image_path, transform=transform_val)
        else:
            # Modo legado: usa ImageFolder (exige subpasta com a imagem)
            val_dataset = datasets.ImageFolder(val_folder, transform_val)

        dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        img_in, _ = next(iter(dataloader))
        args.patch_h = img_in.shape[2]
        args.patch_w = img_in.shape[3]

        # ── Cria máscara de acordo com --mask_type ────────────────────────
        target_mask, target_mask_tensor = create_mask(
            height=args.patch_h,
            width=args.patch_w,
            mask_type=args.mask_type,
        )
        # target_mask       : (1, 1, H, W) BoolTensor
        # target_mask_tensor: (H*W,)       BoolTensor  (flat)
        # ─────────────────────────────────────────────────────────────────

        args.all_pix_num = args.patch_h * args.patch_w
        args.eval_pix_num = args.patch_h * args.patch_w
        print(args)

        folder_path = (
            "./saved/modbase_"
            + str(args.mod_base)
            + "/context_"
            + str(args.context_arm)
            + "_arm_mod_"
            + str(args.dim_arm_mod)
        )
        make_path(folder_path)
        folder_path_ = folder_path + "/sparity_" + str(args.sparsity)
        make_path(folder_path_)
        saved_path = (
            folder_path_
            + "/inr_mod_"
            + str(args.dim_arm_mod)
            + "KODAK_ROI_Global_new_struture_7_"
            + str(args.sythesis_features)
            + "_3_33_orderconcat_operator_"
            + str(args.mod_hid_layer)
            + "_pw_"
            + str(args.lambda_rate)
            + "_img_object"
            + str(it)
            + ".pth"
        )

        # Define total steps
        total_steps = 10
        total_steps_2 = 10
        steps_til_summary = 10

        target_mask_flat = target_mask.flatten()
        if args.use_candidate:
            mask_model = train_with_candidates(
                args, target_mask, target_mask_flat, dataloader
            )
        else:
            mask_model = Masked_INR(
                args,
                target_mask,
                sparsity=args.sparsity,
                in_features=2,
                out_features=3 * args.scale * args.scale,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layer,
            )

        print(mask_model)
        mask_model.cuda()
        print("train the", it, "-th image")

        out_psnr, out_rate, rate_num, decoded_path = train(
            target_mask_flat,
            mask_model,
            dataloader,
            total_steps,
            total_steps_2,
            steps_til_summary,
            it,
            saved_path,
            logger=logger,
            image_name=image_name,
            args=args,
        )

        all_psnr.append(out_psnr)
        all_rate.append(out_rate)
        all_rate_num.append(rate_num)

        # ── Carrega modelo e avalia ───────────────────────────────────────
        checkpoints = torch.load(saved_path)
        mask_model.load_state_dict(checkpoints["model_state_dict"])
        binary_mask = None
        mask_model.cuda()
        mask_model.eval()

        (
            eval_out_psnr,
            eval_ws_ssim,
            eval_y_rate,
            eval_y_rate_num,
            eval_network_rate,
            eval_network_rate_num,
            eval_network_rate_arm,
            eval_network_rate_arm_num,
            eval_network_rate_conv,
            eval_network_rate_conv_num,
        ) = eval_model(target_mask_flat, args, mask_model, binary_mask, dataloader, it)

        eval_all_psnr.append(eval_out_psnr)
        eval_all_y_rate.append(eval_y_rate)
        eval_all_y_rate_num.append(eval_y_rate_num)
        eval_all_mlp_rate.append(eval_network_rate)
        eval_all_mlp_rate_num.append(eval_network_rate_num)
        eval_all_rate_arm.append(eval_network_rate_arm)
        eval_all_rate_arm_num.append(eval_network_rate_arm_num)
        eval_all_rate_conv.append(eval_network_rate_conv)
        eval_all_rate_conv_num.append(eval_network_rate_conv_num)

        if lossyless_path is not None:
            eval_border_rate_num = get_border_bits(lossyless_path, it)
        else:
            # Sem máscara de borda: contribuição de borda é zero
            eval_border_rate_num = 0
        eval_border_rate = eval_border_rate_num / args.eval_pix_num
        eval_all_border_rate.append(eval_border_rate)
        eval_all_border_rate_num.append(eval_border_rate_num)

        eval_total = eval_y_rate + eval_network_rate + eval_border_rate
        eval_total_num = eval_y_rate_num + eval_network_rate_num + eval_border_rate_num
        eval_all_total_rate.append(eval_total)
        eval_all_total_rate_num.append(eval_total_num)

        eval_all_rate_y_mlp_latent = [
            y + mlp + b
            for y, mlp, b in zip(
                eval_all_y_rate, eval_all_mlp_rate, eval_all_border_rate
            )
        ]
        eval_all_rate_y_mlp_latent_num = [
            y + mlp + b
            for y, mlp, b in zip(
                eval_all_y_rate_num, eval_all_mlp_rate_num, eval_all_border_rate_num
            )
        ]

        # ── Log da avaliação completa ─────────────────────────────────────
        logger.log(
            image_name=image_name,
            step="eval",
            image_index=it,
            lambda_rate=args.lambda_rate,
            args=args,
            checkpoint_path=saved_path,
            decoded_image_path=decoded_path,
            metrics={
                "swhdc": 1 if args.use_swhdc else 0,
                "psnr": out_psnr,
                "loss_mse": loss_mse,
                "mask_type": args.mask_type,
                "bits_rate": out_rate,
                "bits_rate_num": rate_num,
                "eval_psnr": eval_out_psnr,
                "eval_ws_ssim": eval_ws_ssim.item()
                if torch.is_tensor(eval_ws_ssim)
                else eval_ws_ssim,  # <--- ADICIONE AQUI
                "eval_y_rate": eval_y_rate,
                "eval_y_rate_num": eval_y_rate_num,
                "eval_network_rate": eval_network_rate,
                "eval_network_rate_num": eval_network_rate_num,
                "eval_arm_rate": eval_network_rate_arm,
                "eval_arm_rate_num": eval_network_rate_arm_num,
                "eval_conv_rate": eval_network_rate_conv,
                "eval_conv_rate_num": eval_network_rate_conv_num,
                "eval_border_rate": eval_border_rate,
                "eval_border_rate_num": eval_border_rate_num,
                "eval_total_rate": eval_total,
                "eval_total_rate_num": eval_total_num,
                # Métricas cumulativas do dataset até agora
                "running_avg_psnr": np.mean(eval_all_psnr),
                "running_avg_total_rate": np.mean(eval_all_rate_y_mlp_latent),
            },
        )
        logger.flush()  # ← grava no CSV após cada imagem
        # ─────────────────────────────────────────────────────────────────

        print(
            "Evaluate the image: PSNR:",
            eval_out_psnr,
            "All bits:",
            eval_all_rate_y_mlp_latent[-1],
            "latent bits:",
            eval_y_rate,
            "network bits:",
            eval_network_rate,
        )
        print(
            "Current eval Ave PSNR:",
            np.mean(eval_all_psnr),
            "Ave Bits",
            np.mean(eval_all_rate_y_mlp_latent),
        )

    # ── Salva listas para prints finais ──────────────────────────────────────
    all_psnr_list_of_lists.append(all_psnr)
    all_rate_list_of_lists.append(all_rate)
    all_rate_num_list_of_lists.append(all_rate_num)
    eval_all_psnr_list_of_lists.append(eval_all_psnr)
    eval_all_y_rate_list_of_lists.append(eval_all_y_rate)
    eval_all_y_rate_num_list_of_lists.append(eval_all_y_rate_num)
    eval_all_mlp_rate_list_of_lists.append(eval_all_mlp_rate)
    eval_all_mlp_rate_num_list_of_lists.append(eval_all_mlp_rate_num)
    eval_all_border_rate_list_of_lists.append(eval_all_border_rate)
    eval_all_border_rate_num_list_of_lists.append(eval_all_border_rate_num)
    eval_all_total_rate_list_of_lists.append(eval_all_total_rate)
    eval_all_total_rate_num_list_of_lists.append(eval_all_total_rate_num)
    eval_all_arm_rate_list_of_lists.append(eval_all_rate_arm)
    eval_all_arm_rate_num_list_of_lists.append(eval_all_rate_arm_num)
    eval_all_conv_rate_list_of_lists.append(eval_all_rate_conv)
    eval_all_conv_rate_num_list_of_lists.append(eval_all_rate_conv_num)

    print("....... Complete all dataset training ......")
    print(
        "Evaluation: Ave Eval PSNR:",
        np.mean(eval_all_psnr),
        "Ave Eval all bits:",
        np.mean(eval_all_rate_y_mlp_latent),
    )

# ── Resultados finais ─────────────────────────────────────────────────────────
print("======== ALL Results ========")
for i, lambda_rate in enumerate(args.lambda_rate_list):
    print(f"Lambda = {lambda_rate}:")
    print("  all_psnr:", all_psnr_list_of_lists[i])
    print("  all_rate:", all_rate_list_of_lists[i])
    print("  eval_all_psnr:", eval_all_psnr_list_of_lists[i])
    print("  eval_all_y_rate:", eval_all_y_rate_list_of_lists[i])
    print("  eval_all_mlp_rate:", eval_all_mlp_rate_list_of_lists[i])
    print("  eval_all_border_rate:", eval_all_border_rate_list_of_lists[i])
    print("  eval_all_total_rate:", eval_all_total_rate_list_of_lists[i])
    print("---------------------------------")

print(f"\n[Done] Resultados salvos em: {logger.csv_path}")
print(f"[Done] Imagens decodificadas em: {logger.images_dir}")
