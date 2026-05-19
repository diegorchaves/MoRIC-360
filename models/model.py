import math
from itertools import islice
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch import Tensor, index_select, nn

from enc.utils.misc import (
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
from utils.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from utils.quantizer import quantize
from utils.upsampling import Upsampling


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1])
                )
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class GetSubnet_batch(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        batch_size, w1, w2 = scores.shape
        score_reshape = scores.view(batch_size, -1)
        _, indices = torch.sort(score_reshape, dim=1, descending=True)
        j = int((1 - k) * score_reshape.size(1))
        binary_mask = torch.zeros_like(score_reshape)
        binary_mask.scatter_(1, indices[:, :j], 1)
        binary_mask = binary_mask.view(batch_size, w1, w2)
        return binary_mask

    @staticmethod
    def backward(ctx, g):
        return g, None


class NonAffineBatchNorm(nn.BatchNorm1d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


# =============================================================================
# SWHDC — Spherically-aware Weighted Horizontal Dilated Convolution
# Substitui nn.Conv2d nos ramos net e global_net quando args.use_swhdc=True.
# Aplica padding circular na horizontal e reflect na vertical internamente,
# por isso SynthesisLayer deve desativar seu próprio padding ao usar SWHDC.
# =============================================================================
class SWHDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(SWHDC, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        )

    def forward(self, x):
        _, _, h, w = x.shape
        N = len(self.dilations)
        phi = (torch.linspace(0, 1, h, device=x.device)) * torch.pi
        Rs = torch.min(
            torch.tensor(N, device=x.device, dtype=torch.float32),
            torch.abs(
                torch.ones(1, device=x.device)
                / torch.sin(phi - torch.finfo(torch.float32).eps)
            ),
        )

        dilations_tensor = torch.tensor(
            self.dilations, device=x.device, dtype=torch.float32
        ).view(N, 1)
        Rs_expanded = Rs.unsqueeze(0)

        cR = torch.ceil(Rs_expanded)
        fR = torch.floor(Rs_expanded)

        mask_exact = dilations_tensor == Rs_expanded
        mask_floor = (dilations_tensor == fR) & ~mask_exact
        mask_ceil = (dilations_tensor == cR) & ~mask_exact & ~mask_floor

        row_wise_weights = torch.zeros(N, h, device=x.device)
        row_wise_weights = torch.where(
            mask_exact, torch.ones_like(row_wise_weights), row_wise_weights
        )
        row_wise_weights = torch.where(mask_floor, cR - Rs_expanded, row_wise_weights)
        row_wise_weights = torch.where(mask_ceil, Rs_expanded - fR, row_wise_weights)

        outputs = []
        for idx in range(N):
            dilation_rate = self.dilations[idx]
            v_padding_dilation = 1 * (self.conv.kernel_size[0] - 1) // 2
            h_padding_dilation = dilation_rate * (self.conv.kernel_size[0] - 1) // 2
            h_padding_tuple = (h_padding_dilation, h_padding_dilation, 0, 0)
            v_padding_tuple = (0, 0, v_padding_dilation, v_padding_dilation)

            x2 = F.pad(x, h_padding_tuple, mode="circular")
            x2 = F.pad(x2, v_padding_tuple, mode="reflect")

            out = F.conv2d(
                x2,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                dilation=(1, dilation_rate),
                groups=self.conv.groups,
            )

            out = torch.einsum("c,abcd->abcd", row_wise_weights[idx], out)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)  # [N, B, C, H, W]
        return torch.sum(outputs, dim=0)  # [B, C, H, W]


# =============================================================================
# SynthesisLayer — aceita custom_conv opcional (ex: SWHDC).
# Quando custom_conv é fornecido:
#   - self.pad vira nn.Identity (SWHDC já faz o próprio padding)
#   - a inicialização de peso usa .conv.weight em vez de .weight
# =============================================================================
class SynthesisLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity(),
        custom_conv=None,
    ):
        super().__init__()

        if custom_conv is not None:
            # SWHDC gerencia seu próprio padding — desativa o pad externo
            self.pad = nn.Identity()
            self.conv_layer = custom_conv
            with torch.no_grad():
                self.conv_layer.conv.weight.data = (
                    self.conv_layer.conv.weight.data / output_ft**2
                )
                self.conv_layer.conv.bias.data = self.conv_layer.conv.bias.data * 0.0
        else:
            self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
            self.conv_layer = nn.Conv2d(input_ft, output_ft, kernel_size)
            with torch.no_grad():
                self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft**2
                self.conv_layer.bias.data = self.conv_layer.bias.data * 0.0

        self.non_linearity = non_linearity

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)))


class SynthesisResidualLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity(),
    ):
        super().__init__()

        assert input_ft == output_ft, (
            f"Residual layer in/out dim must match. Input = {input_ft}, output = {output_ft}"
        )

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(input_ft, output_ft, kernel_size)
        self.non_linearity = non_linearity

        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.0
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.0

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)) + x)


class ModConv(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, mod_layer):
        super().__init__()
        self.residual = False
        self.hid_channels = hid_channels
        self.hid_layer = mod_layer

        self.conv1_1 = SynthesisLayer(in_channels, hid_channels, 1, nn.GELU())
        self.conv1_2 = SynthesisLayer(hid_channels, 3, 1, nn.GELU())
        self.conv2_1 = SynthesisResidualLayer(3, 3, 3, nn.GELU())
        self.conv2_2 = SynthesisResidualLayer(3, 3, 3)

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

    def forward(self, x):
        out_0 = self.conv1_1(x)
        out_1 = self.conv1_2(out_0)
        out_2 = self.conv2_1(out_1)
        out_3 = self.conv2_2(out_2)
        return out_3


# =============================================================================
# LocallyConnectedBlock
# Parâmetros novos:
#   use_swhdc  (bool)  — se True, primeira camada de net e global_net usa SWHDC
#   dilations  (list)  — lista de dilatações para SWHDC; obrigatório se use_swhdc=True
#
# Quando use_swhdc=False o comportamento é idêntico ao original (kernel_size=1).
# Quando use_swhdc=True  a primeira camada passa para kernel_size=3 com SWHDC,
# que é onde a dilatação adaptativa por latitude agrega valor. As camadas
# residuais seguintes continuam com kernel=1 e nn.Conv2d convencional.
# =============================================================================
class LocallyConnectedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        global_hid_channels,
        local_hid_channels,
        out_channels,
        mod_layer,
        use_swhdc: bool = False,
        dilations: list = None,
    ):
        super().__init__()

        if use_swhdc:
            assert dilations is not None, (
                "dilations é obrigatório quando use_swhdc=True"
            )

        def make_first_layer(in_ch, out_ch):
            """Primeira camada do ramo: SWHDC ou conv 1×1 convencional."""
            if use_swhdc:
                return SynthesisLayer(
                    in_ch,
                    out_ch,
                    3,
                    nn.GELU(),
                    custom_conv=SWHDC(in_ch, out_ch, 3, dilations),
                )
            else:
                return SynthesisLayer(in_ch, out_ch, 1, nn.GELU())

        # --- ramo net (objeto / equador) ---
        self.net = nn.Sequential(
            make_first_layer(2, local_hid_channels),
            SynthesisResidualLayer(
                local_hid_channels, local_hid_channels, 1, nn.GELU()
            ),
            SynthesisResidualLayer(
                local_hid_channels, local_hid_channels, 1, nn.GELU()
            ),
            SynthesisResidualLayer(local_hid_channels, 3, 1),
        )

        # --- ramo global_net (fundo / polos) ---
        self.global_net = nn.Sequential(
            make_first_layer(2, local_hid_channels),
            SynthesisResidualLayer(
                local_hid_channels, local_hid_channels, 1, nn.GELU()
            ),
            SynthesisResidualLayer(
                local_hid_channels, local_hid_channels, 1, nn.GELU()
            ),
            SynthesisResidualLayer(local_hid_channels, 3, 1),
        )

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

    def forward(self, x, y):
        output_local = self.net(x)
        output_global = self.global_net(y)
        return output_local, output_global


class LocalGlobalBlock(LocallyConnectedBlock):
    def __init__(
        self,
        in_channels,
        global_hid_channels,
        local_hid_channels,
        out_channels,
        mod_layer,
        mask,
        use_swhdc: bool = False,
        dilations: list = None,
    ):
        super().__init__(
            in_channels,
            global_hid_channels,
            local_hid_channels,
            out_channels,
            mod_layer,
            use_swhdc=use_swhdc,
            dilations=dilations,
        )

        self.mask = mask

        self.agg_func = nn.Sequential(
            SynthesisLayer(global_hid_channels + 6, 3, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels + 9, 3, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels + 12, 3, 1, nn.GELU()),
        )

        self.full_net = nn.Sequential(
            SynthesisLayer(in_channels, global_hid_channels, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels, 3, 1, nn.GELU()),
            SynthesisResidualLayer(3, 3, 3, nn.GELU()),
            SynthesisResidualLayer(3, 3, 3, nn.GELU()),
        )

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

    def forward(self, coordinate, combined_latent):
        self.mask = self.mask.bool()

        object_latent = torch.zeros_like(coordinate)
        background_latent = torch.zeros_like(coordinate)
        object_latent[self.mask.expand_as(coordinate)] = coordinate[
            self.mask.expand_as(coordinate)
        ]
        background_latent[~self.mask.expand_as(coordinate)] = coordinate[
            ~self.mask.expand_as(coordinate)
        ]

        all_outputs = []
        out_full = []

        global_layer_input = background_latent
        local_layer_input = object_latent
        full_layer_input = combined_latent
        id = 0

        for full_layer in self.full_net:
            full_layer_input = full_layer(full_layer_input)
            all_outputs.append(full_layer_input)

        out_full.append(torch.cat(all_outputs[:2], dim=1))
        out_full.append(torch.cat(all_outputs[:3], dim=1))
        out_full.append(torch.cat(all_outputs, dim=1))

        for local_layer, global_layer, agg_layer in zip(
            self.net, self.global_net, self.agg_func
        ):
            local_layer_input = local_layer(local_layer_input)
            global_layer_input = global_layer(global_layer_input)

            if id < 3:
                device = out_full[id].device
                self.mask = self.mask.to(device)
                # Nota: squeeze() converte (1,1,H,W) → (H,W); verificar shape se
                # batch_size > 1 ou se a máscara tiver dimensões diferentes.
                self.mask = self.mask.squeeze()

                output_full_local = torch.where(
                    self.mask,
                    out_full[id],
                    torch.zeros_like(out_full[id], device=device),
                )
                output_full_global = torch.where(
                    ~self.mask,
                    out_full[id],
                    torch.zeros_like(out_full[id], device=device),
                )
                global_layer_input = agg_layer(
                    torch.cat([global_layer_input, output_full_global], dim=-3)
                )
                local_layer_input = agg_layer(
                    torch.cat([local_layer_input, output_full_local], dim=-3)
                )

            id += 1

        global_layer_input = self.global_net[-1](global_layer_input)
        local_layer_input = self.net[-1](local_layer_input)

        return local_layer_input, global_layer_input


class Masked_INR(nn.Module):
    def __init__(
        self,
        args,
        target_mask,
        sparsity,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
    ):
        super().__init__()
        self.sparsity = sparsity
        self.net = []

        self.h = target_mask.shape[-2]
        self.w = target_mask.shape[-1]

        self.target_mask = target_mask
        self.pe_flag = 0
        if self.pe_flag == 1:
            self.pe = PosEncodingNeRF(2, (self.h, self.w))
            input_dim = 30
        else:
            input_dim = 2

        self.upsampling_2d = Upsampling(
            args.local_upsampling_kernel_size,
            args.static_upsampling_kernel,
            args.highest_flag,
        )

        self.dim_arm = args.dim_arm_mod
        self.n_hidden_layers_arm = 2
        self.arm = Arm(args.context_arm, args.dim_arm_mod, self.n_hidden_layers_arm)

        self.quantizer_type = "softround"
        self.quantizer_noise_type = "kumaraswamy"
        self.soft_round_temperature = 0.3
        self.noise_parameter = 2.0
        max_mask_size = 9

        self.modulation_base_number = args.mod_base

        self.fact_shape = []
        if args.highest_flag == 1:
            for i in range(self.modulation_base_number):
                self.fact_shape.append((self.h // (2**i), self.w // (2**i)))
        else:
            for i in range(self.modulation_base_number):
                self.fact_shape.append(
                    (self.h // (2 ** (i + 1)), self.w // (2 ** (i + 1)))
                )
        self.fact_shape.reverse()

        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.dim_arm}"
        )

        self.mask_size = 9
        self.encoder_gains_sf = 16
        print("Quantizer parameter: encoding gain ", self.encoder_gains_sf)

        self.all_pix_num = self.h * self.w // args.scale // args.scale
        print("total pixel:", self.all_pix_num)

        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(args.context_arm),
            persistent=False,
        )

        self.latent_factor = args.latent_factor

        # ------------------------------------------------------------------
        # Instanciação do LocalGlobalBlock com suporte a SWHDC.
        # args.use_swhdc   (bool) — ativado via --use_swhdc na linha de comando
        # args.swhdc_dilations (list[int]) — ex: --swhdc_dilations 1 2 3 4
        # ------------------------------------------------------------------
        self.conv_mod = LocalGlobalBlock(
            in_channels=self.modulation_base_number,
            global_hid_channels=args.sythesis_features,
            local_hid_channels=3,
            out_channels=hidden_layers + 1,
            mod_layer=args.mod_hid_layer,
            mask=self.target_mask,
            use_swhdc=args.use_swhdc,
            dilations=args.swhdc_dilations if args.use_swhdc else None,
        )

        self.modules_to_send = ["arm", "conv_mod", "upsampling_2d"]

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.modulation_sf = nn.ParameterList()

        self.mask_sf = []
        for layer_idx in range(self.modulation_base_number):
            mod_shape = self.fact_shape[layer_idx]
            shits = nn.Parameter(
                torch.zeros(args.batch_size, 1, mod_shape[0], mod_shape[1])
            ).cuda()
            if layer_idx > 0:
                masks = F.max_pool2d(target_mask.float(), kernel_size=2)
            else:
                masks = target_mask
            target_mask = masks
            self.mask_sf.append(masks.cuda())
            self.modulation_sf.append(shits)
            print("Get Mod with shape", shits.shape, "at layer:", layer_idx + 1)

    def quantize_all_latent(self, latent, coords):
        q_shifts_all = []
        q_shifts_all_for_conv = []
        q_shifts_all_for_conv_o = []
        q_shifts_all_for_conv_b = []

        for id in range(len(latent)):
            q_shifts_id = quantize(
                latent[id] * self.encoder_gains_sf,
                self.quantizer_noise_type if self.training else "none",
                self.quantizer_type if self.training else "hardround",
                self.soft_round_temperature,
                self.noise_parameter,
            )

            q_shifts_all_for_conv_o.append(
                q_shifts_id * self.mask_sf[len(latent) - id - 1]
            )
            q_shifts_all_for_conv_b.append(
                q_shifts_id * (~self.mask_sf[len(latent) - id - 1].bool())
            )
            q_shifts_all_for_conv.append(q_shifts_id)

        q_upsample_conv = self.upsampling_2d(q_shifts_all_for_conv, self.mask_sf)

        weight_shift_all, weight_shift_all_b = self.conv_mod(coords, q_upsample_conv)

        return q_shifts_all_for_conv, weight_shift_all, weight_shift_all_b

    def get_param(self):
        param = OrderedDict()
        param.update({f"conv_mod.{k}": v for k, v in self.conv_mod.get_param().items()})
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        param.update(
            {f"upsampling_2d.{k}": v for k, v in self.upsampling_2d.get_param().items()}
        )
        param.update(
            {f"modulation_sf.{i}": v for i, v in enumerate(self.modulation_sf)}
        )
        return param

    def set_param(self, param):
        conv_mod_param = {
            k[len("conv_mod.") :]: v
            for k, v in param.items()
            if k.startswith("conv_mod.")
        }
        arm_param = {
            k[len("arm.") :]: v for k, v in param.items() if k.startswith("arm.")
        }
        upsampling_param = {
            k[len("upsampling_2d.") :]: v
            for k, v in param.items()
            if k.startswith("upsampling_2d.")
        }

        self.conv_mod.set_param(conv_mod_param)
        self.arm.set_param(arm_param)
        self.upsampling_2d.set_param(upsampling_param)

        modulation_sf_param = {
            int(k.split(".")[1]): v
            for k, v in param.items()
            if k.startswith("modulation_sf.")
        }
        for i, v in modulation_sf_param.items():
            self.modulation_sf[i].data.copy_(v.data)

    def estimate_rate(self, decoder_side_latent, arm_model):
        flat_context = torch.cat(
            [
                _get_neighbor(
                    spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index
                )
                for i, spatial_latent_i in enumerate(decoder_side_latent)
            ],
            dim=0,
        )
        flat_latent = torch.cat(
            [
                spatial_latent_i.view(-1)
                for i, spatial_latent_i in enumerate(decoder_side_latent)
            ],
            dim=0,
        )
        flat_context_in = flat_context.unsqueeze(0).transpose(1, 2)
        flat_mu, flat_scale, flat_log_scale__ = arm_model(flat_context_in)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,
        )
        flat_rate = -torch.log2(proba)
        return flat_rate

    def get_network_rate(self):
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }
        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )
        return rate_per_module

    def compute_rate(self):
        all_score_list = []
        for layer_id, layer in enumerate(self.net):
            all_score_list.append(layer.scores.view(-1))
        all_score = torch.cat(all_score_list, dim=0)
        num_top_20_percent = int(len(all_score) * (1 - self.sparsity))
        topk_values, _ = torch.topk(all_score, num_top_20_percent)
        threshold = topk_values.min().item()
        out_num = []
        for k in range(len(all_score_list)):
            out_num.append(torch.sum(all_score_list[k] >= threshold).item())
        return out_num

    def forward(self, coords, in_mask=None):
        saved_mask = []
        if self.pe_flag == 1:
            input_ = self.pe(coords)
        else:
            input_ = coords

        q_shifts_all_viewed, weighted_q_shift_all, weight_shift_all_b = (
            self.quantize_all_latent(self.modulation_sf, coords)
        )

        input_ = weighted_q_shift_all
        input_1 = weight_shift_all_b
        mask_b = ~self.mask_sf[0]

        flat_rate = self.estimate_rate(q_shifts_all_viewed, self.arm)
        batch_size = input_.shape[0]

        input_ = input_.view(batch_size, 3, -1)[:, :, self.mask_sf[0].flatten().bool()]
        input_1 = input_1.view(batch_size, 3, -1)[:, :, mask_b.flatten().bool()]

        total_length = self.h * self.w
        concatenated_input = torch.zeros(
            batch_size, 3, total_length, device=input_.device
        )
        concatenated_input[:, :, self.mask_sf[0].flatten().bool()] = input_
        concatenated_input[:, :, mask_b.flatten().bool()] = input_1
        return concatenated_input.permute(0, 2, 1), flat_rate, saved_mask
