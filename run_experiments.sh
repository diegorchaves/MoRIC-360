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

set -uo pipefail

# ── Configurações principais ─────────────────────────────────────────────────

IMAGES_DIR="${IMAGES_DIR:-./ctc}"       # pasta com as imagens a treinar
RESULTS_DIR="${RESULTS_DIR:-./results_ctc_5080}"  # onde salvar CSVs e imagens decodificadas
LOGS_DIR="${LOGS_DIR:-./logs_5080}"           # onde salvar os .out de cada cenário

PYTHON="${PYTHON:-python}"

# ── Dimensões do experimento ─────────────────────────────────────────────────

# 1) Lambdas
LAMBDAS=(
    #"1e-4"
    #"1.2e-3"
    #"2.3e-3"
    #"3.4e-3"
    #"4.5e-3"
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
    for swhdc_entry in "${SWHDC_CONFIGS[@]}"; do
        swhdc_label="${swhdc_entry%%|*}"
        swhdc_args_str="${swhdc_entry##*|}"

        if [[ -n "${swhdc_args_str}" ]]; then
            read -ra swhdc_args <<< "${swhdc_args_str}"
        else
            swhdc_args=()
        fi

        for loss_type in "${LOSS_TYPES[@]}"; do
            for mask_type in "${MASK_TYPES[@]}"; do
                run=$(( run + 1 ))
                tag="lambda${lambda}_${swhdc_label}_mask${mask_type}_loss${loss_type}"
                log="${LOGS_DIR}/train_${tag}.out"

                echo ""
                echo "────────────────────────────────────────────────────────"
                echo "  Cenário ${run}/${total}: ${tag}"
                echo "  log → ${log}"
                echo "────────────────────────────────────────────────────────"

                # Pula se já foi concluído com sucesso
                if [[ -f "${log}" ]] && grep -q "\[OK\] cenário" "${log}"; then
                    echo "  [SKIP] cenário ${run}/${total} já concluído."
                    continue
                fi

                $PYTHON -u train.py \
                    "${BASE_ARGS[@]}"              \
                    --lambda_rate_list "${lambda}" \
                    --mask_type        "${mask_type}" \
                    --run_tag          "${tag}"    \
                    --loss_type        "${loss_type}" \
                    ${swhdc_args_str}              \
                    > "${log}" 2>&1 || true

                # Unificado: verifica e grava no log
                if grep -q "Traceback\|CUDA out of memory\|Killed" "${log}" 2>/dev/null; then
                    echo "  [ERRO] cenário ${run}/${total} falhou — veja ${log}"
                else
                    echo "[OK] cenário ${run}/${total} concluído." | tee -a "${log}"
                fi

                nvidia-smi --gpu-reset -i 0 || true
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
