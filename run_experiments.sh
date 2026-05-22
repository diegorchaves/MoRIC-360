#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
#
# Executa train.py para cada combinação de:
#   lambdas  × swhdc (on/off) × mask_type
#
# Total de cenários: 3 lambdas × 2 swhdc × 2 máscaras = 12 runs
#
# Uso:
#   bash run_experiments.sh
#   IMAGES_DIR=/outro/caminho bash run_experiments.sh
# =============================================================================

set -euo pipefail

# ── Configurações principais ─────────────────────────────────────────────────

IMAGES_DIR="${IMAGES_DIR:-./ctc}"       # pasta com as imagens a treinar
RESULTS_DIR="${RESULTS_DIR:-./results_ctc_3}"  # onde salvar CSVs e imagens decodificadas
LOGS_DIR="${LOGS_DIR:-./logs_3}"           # onde salvar os .out de cada cenário

PYTHON="${PYTHON:-python}"

# ── Dimensões do experimento ─────────────────────────────────────────────────

# 1) Lambdas
LAMBDAS=(
    "1e-4"
    "1.2e-3"
    "2.3e-3"
    "3.4e-3"
    "4.5e-3"
    "5.6e-3"
    "6.7e-3"
    "7.8e-3"
    "8.9e-3"
    "1e-2"
)

# 2) Configurações de SWHDC
#    Cada entrada é "label|args_extras"
#    label  → usado no run_tag e no nome do log
#    args   → flags adicionais repassadas ao train.py (vazio = sem swhdc)
SWHDC_CONFIGS=(
    "swhdc|--use_swhdc --swhdc_dilations 1 2 3 4"
    "noswhdc|"
)

# 3) Tipos de máscara
MASK_TYPES=(
    "full"
    "erp"
)

LOSS_TYPES=(
    "wsmse"
    "mse"
)

# ── Argumentos fixos (comuns a todos os cenários) ────────────────────────────
BASE_ARGS=(
    --images_dir "${IMAGES_DIR}"
    --results_dir "${RESULTS_DIR}"
)

# ── Setup ────────────────────────────────────────────────────────────────────

mkdir -p "${LOGS_DIR}"

total=$(( ${#LAMBDAS[@]} * ${#SWHDC_CONFIGS[@]} * ${#MASK_TYPES[@]} *  ${#LOSS_TYPES[@]} ))
run=0

echo "================================================================"
echo "  Iniciando experimentos: ${total} cenários"
echo "  images_dir : ${IMAGES_DIR}"
echo "  results_dir: ${RESULTS_DIR}"
echo "  logs_dir   : ${LOGS_DIR}"
echo "================================================================"

# ── Loop principal ───────────────────────────────────────────────────────────

for lambda in "${LAMBDAS[@]}"; do
    sleep 5s
    for swhdc_entry in "${SWHDC_CONFIGS[@]}"; do

        sleep 5s
        swhdc_label="${swhdc_entry%%|*}"          # tudo antes do "|"
        swhdc_args_str="${swhdc_entry##*|}"        # tudo depois do "|"

        # Converte string de args em array (lida com args vazios)
        if [[ -n "${swhdc_args_str}" ]]; then
            read -ra swhdc_args <<< "${swhdc_args_str}"
        else
            swhdc_args=()
        fi

        for loss_type in "${LOSS_TYPES[@]}"; do
            sleep 5s
            for mask_type in "${MASK_TYPES[@]}"; do
                sleep 5s
                run=$(( run + 1 ))
                tag="lambda${lambda}_${swhdc_label}_mask${mask_type}_loss${loss_type}"
                log="${LOGS_DIR}/train_${tag}.out"

                echo ""
                echo "────────────────────────────────────────────────────────"
                echo "  Cenário ${run}/${total}: ${tag}"
                echo "  log → ${log}"
                echo "────────────────────────────────────────────────────────"

                $PYTHON -u train.py \
                    "${BASE_ARGS[@]}"            \
                    --lambda_rate_list "${lambda}" \
                    --mask_type        "${mask_type}" \
                    --run_tag          "${tag}"    \
                    --loss_type        "${loss_type}" \
                    ${swhdc_args_str}            \
                    > "${log}" 2>&1

                echo "  [OK] cenário ${run}/${total} concluído."
                nvidia-smi --gpu-reset -i 0
            done
        done
    done
done

echo ""
echo "================================================================"
echo "  Todos os ${total} cenários concluídos."
echo "  Resultados em : ${RESULTS_DIR}"
echo "  Logs em       : ${LOGS_DIR}"
echo "================================================================"
