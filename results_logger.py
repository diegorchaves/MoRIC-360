"""
results_logger.py
=================
Módulo para salvar resultados e imagens decodificadas do codec INR.

Uso básico:
    from results_logger import ResultsLogger

    logger = ResultsLogger(base_dir="./results")
    logger.log(image_name="kodim01", step="eval", metrics={...})
    logger.save_decoded_image(tensor, image_name="kodim01", lambda_rate=1e-3)
    logger.flush()  # salva CSV
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_python(value: Any) -> Any:
    """Converte tipos numpy/torch para tipos nativos Python (serializáveis)."""
    if isinstance(value, (torch.Tensor,)):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _make_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# ResultsLogger
# ---------------------------------------------------------------------------


class ResultsLogger:
    """
    Registra métricas em CSV e salva imagens decodificadas de forma organizada.

    Estrutura de diretórios gerada:
        base_dir/
        ├── results.csv                      ← todas as métricas
        ├── decoded_images/
        │   ├── lambda_1e-03/
        │   │   ├── kodim01_step-eval.png
        │   │   └── kodim02_step-eval.png
        │   └── lambda_1e-04/
        │       └── ...
        └── checkpoints/                     ← (gerenciado externamente, só referenciado)

    Parâmetros
    ----------
    base_dir : str
        Diretório raiz onde tudo será salvo.
    csv_filename : str
        Nome do arquivo CSV principal.
    extra_columns : list[str], opcional
        Colunas extras que você queira garantir na ordem do cabeçalho.
        Novas colunas descobertas automaticamente são sempre adicionadas ao fim.
    run_tag : str, opcional
        Tag livre para identificar o experimento (ex: "ablation_v2").
        Salva em todas as linhas como coluna `run_tag`.
    """

    # Colunas que SEMPRE aparecem primeiro (na ordem abaixo).
    BASE_COLUMNS = [
        "timestamp",
        "run_tag",
        "image_name",
        "image_index",
        "lambda_rate",
        "step",  # "train_stage1" | "train_stage2" | "eval"
        # ---- modelo ----
        "model_arch",
        "context_arm",
        "dim_arm_mod",
        "synthesis_features",
        "hidden_features",
        "hidden_layers",
        "mod_base",
        "scale",
        "mask_type",
        "swhdc_tag",
        "wsmse_tag",
        # ---- métricas principais ----
        "psnr",
        "loss_mse",
        "bits_rate",  # bpp normalizado por pixel
        "bits_rate_num",  # bits totais absolutos
        # ---- métricas de avaliação (eval_model) ----
        "eval_psnr",
        "eval_y_rate",
        "eval_y_rate_num",
        "eval_network_rate",
        "eval_network_rate_num",
        "eval_arm_rate",
        "eval_arm_rate_num",
        "eval_conv_rate",
        "eval_conv_rate_num",
        "eval_border_rate",
        "eval_border_rate_num",
        "eval_total_rate",
        "eval_total_rate_num",
        # ---- info do arquivo salvo ----
        "checkpoint_path",
        "decoded_image_path",
    ]

    def __init__(
        self,
        base_dir: str = "./results",
        csv_filename: str = "results.csv",
        extra_columns: list = None,
        run_tag: str = "",
    ):
        self.base_dir = _make_dir(base_dir)
        self.images_dir = _make_dir(os.path.join(base_dir, "decoded_images"))
        self.csv_path = os.path.join(base_dir, csv_filename)
        self.run_tag = run_tag

        # Colunas conhecidas (ordem importa para o cabeçalho)
        self._columns: list[str] = list(self.BASE_COLUMNS)
        if extra_columns:
            for col in extra_columns:
                if col not in self._columns:
                    self._columns.append(col)

        self._rows: list[dict] = []
        self._existing_rows: list[dict] = self._load_existing_csv()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def log(
        self,
        image_name: str,
        step: str,
        metrics: dict,
        args=None,
        image_index: int = None,
        lambda_rate: float = None,
        checkpoint_path: str = None,
        decoded_image_path: str = None,
    ) -> dict:
        """
        Registra uma linha de métricas.

        Parâmetros
        ----------
        image_name : str
            Ex: "kodim01"
        step : str
            Uma das etapas: "train_stage1", "train_stage2", "eval", etc.
        metrics : dict
            Dicionário livre com qualquer métrica. Novas chaves são adicionadas
            automaticamente ao CSV sem nenhuma alteração de código.
        args : argparse.Namespace, opcional
            Se passado, extrai automaticamente os hiperparâmetros do modelo.
        image_index : int, opcional
            Índice numérico da imagem no dataset.
        lambda_rate : float, opcional
            Lambda de taxa-distorção usado neste experimento.
        checkpoint_path : str, opcional
            Caminho do checkpoint salvo.
        decoded_image_path : str, opcional
            Caminho da imagem decodificada salva.

        Retorna
        -------
        dict : a linha registrada (útil para debug).
        """
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_tag": self.run_tag,
            "image_name": image_name,
            "image_index": image_index,
            "lambda_rate": _to_python(lambda_rate),
            "step": step,
            "checkpoint_path": checkpoint_path or "",
            "decoded_image_path": decoded_image_path or "",
        }

        # Hiperparâmetros do modelo a partir de args
        if args is not None:
            row.update(
                {
                    "model_arch": "Masked_INR",
                    "context_arm": getattr(args, "context_arm", None),
                    "dim_arm_mod": getattr(args, "dim_arm_mod", None),
                    "synthesis_features": getattr(args, "sythesis_features", None),
                    "hidden_features": getattr(args, "hidden_features", None),
                    "hidden_layers": getattr(args, "hidden_layer", None),
                    "mod_base": getattr(args, "mod_base", None),
                    "scale": getattr(args, "scale", None),
                }
            )

        # Métricas passadas livremente
        for k, v in metrics.items():
            row[k] = _to_python(v)

        # Descoberta automática de novas colunas
        for key in row:
            if key not in self._columns:
                self._columns.append(key)

        self._rows.append(row)
        return row

    def save_decoded_image(
        self,
        image_tensor: torch.Tensor,
        image_name: str,
        lambda_rate: float,
        step: str = "eval",
        suffix: str = "",
        pos_suffix: str = "",
    ) -> str:
        """
        Salva uma imagem decodificada em PNG.

        Parâmetros
        ----------
        image_tensor : torch.Tensor
            Shape (1, H, W, 3) ou (H, W, 3) ou (3, H, W), valores em [0, 1].
        image_name : str
            Ex: "kodim01"
        lambda_rate : float
            Usado para organizar subpastas.
        step : str
            Usado no nome do arquivo.
        suffix : str
            Sufixo livre para o nome do arquivo.

        Retorna
        -------
        str : caminho completo da imagem salva.
        """
        try:
            import torchvision.utils as vutils
        except ImportError:
            raise ImportError("torchvision é necessário para salvar imagens.")

        # Subpasta por lambda
        lambda_str = f"lambda_{lambda_rate:.0e}".replace("+", "")
        subdir = _make_dir(os.path.join(self.images_dir, lambda_str))

        # Nome do arquivo
        parts = [image_name, f"step-{step}"]
        if suffix:
            parts.append(suffix)
        if pos_suffix:
            parts.append(str(pos_suffix))
        filename = "_".join(parts) + ".png"
        out_path = os.path.join(subdir, filename)

        # Normaliza tensor para (C, H, W) float [0,1]
        t = image_tensor.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)  # (1,H,W,3) → (H,W,3)
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)  # (H,W,3) → (3,H,W)
        t = t.clamp(0.0, 1.0)

        vutils.save_image(t, out_path)
        return out_path

    def flush(self) -> str:
        """
        Grava todos os registros pendentes no CSV.
        Adiciona ao arquivo existente se ele já tiver conteúdo.

        Retorna
        -------
        str : caminho do CSV salvo.
        """
        all_rows = self._existing_rows + self._rows

        # Garante que todas as colunas existentes no arquivo sejam preservadas
        for row in all_rows:
            for key in row:
                if key not in self._columns:
                    self._columns.append(key)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k, "") for k in self._columns})

        print(
            f"[ResultsLogger] CSV salvo em: {self.csv_path}  ({len(self._rows)} novas linhas)"
        )
        self._existing_rows = all_rows
        self._rows = []
        return self.csv_path

    def log_and_flush(self, **kwargs) -> str:
        """Atalho: log() + flush() em uma chamada."""
        self.log(**kwargs)
        return self.flush()

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _load_existing_csv(self) -> list[dict]:
        """Lê linhas já existentes no CSV para não sobrescrever ao dar flush."""
        if not os.path.exists(self.csv_path):
            return []
        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                for key in row:
                    if key not in self._columns:
                        self._columns.append(key)
        return rows

    def summary(self) -> dict:
        """Retorna médias das métricas numéricas dos registros pendentes."""
        if not self._rows:
            return {}
        keys = [
            k
            for k in self._rows[0]
            if k
            not in (
                "timestamp",
                "run_tag",
                "image_name",
                "step",
                "checkpoint_path",
                "decoded_image_path",
            )
        ]
        summary = {}
        for k in keys:
            vals = []
            for row in self._rows:
                v = row.get(k)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            if vals:
                summary[k] = {
                    "mean": np.mean(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                }
        return summary
