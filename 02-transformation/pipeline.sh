FILES=$(find ../01-dataset-raw -type f -name "*.csv")
OUTPUT_DIR="../02-transformation/02.1-dataset-processed"
PYTHON=$(realpath ../.venv/bin/python)
LOG_FILE="manager.log"

log() {
    local level="$1"
    shift

    local message="$(date '+%Y-%m-%d %H:%M:%S,%3N') $level $*"

    echo "$message" >> "$LOG_FILE"
}

time (
    # Número de processos paralelos = CPUs - 1
    MAX_JOBS=$(( $(nproc) - 1 ))

    # Garante pelo menos 1 processo
    [ $MAX_JOBS -lt 1 ] && MAX_JOBS=1

    log DEBUG "Executando com $MAX_JOBS processos paralelos"

    for window_length in $(seq 60 60 900); do
        output_dir_window="$OUTPUT_DIR/$window_length"
        mkdir -p "$output_dir_window"
        log DEBUG "Current Window Length $window_length in $output_dir_window ..."
        time (
            for file in $FILES; do
                # Espera enquanto houver muitos processos em execução
                while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
                    sleep 0.2
                done

                (
                    # Gerar o caminho parcial para manter a estrutura de diretórios e adicionar o window_length
                    path_parcial=$(echo "$file" | sed 's|../01-dataset-raw/||')
                    dir_parcial=$(dirname "$path_parcial")

                    mkdir -p "$output_dir_window/$dir_parcial"

                    $PYTHON process_data.py \
                        --input $file \
                        --output $OUTPUT_DIR/$path_parcial \
                        --window_length $window_length \
                        --embedding_dimension 4 
                ) &
            done
        )
        log DEBUG "All processes with window length $window_length have finished."
    done
)
