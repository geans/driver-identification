import argparse
import logging
import time

# import warnings
from pathlib import Path

import ordpy
import pandas as pd

# Suprimir o aviso específico
# warnings.filterwarnings("ignore", message="Be mindful the correct calculation of Fisher information depends on all possible permutations")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
)

# Arquivo
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# # Terminal
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)



def process_data(
        input_path: Path, 
        output_path: Path, 
        window_length: int, 
        embedding_dimension: int
    ):
    output_path_time = output_path.with_suffix('.time.csv')

    exec_time = time.time()
    # Exemplo de leitura
    if input_path.exists():
        df = pd.read_csv(input_path)

        new_df = None
        new_df_sz = 0
        time_hc = []
        time_fs = []
        # Processamento dos dados
        total_windows = len(df) - window_length + 1
        logger.debug(f"Processando {total_windows} janelas de dados com tamanho {window_length} e dimensão de incorporação {embedding_dimension}...")
        for i in range(total_windows):
            window = df.iloc[i:i + window_length]
            row = {}
            for feature in window.columns:
                t0 = time.time()
                data_probs = ordpy.ordinal_distribution(
                    window, 
                    dx=embedding_dimension, 
                    return_missing=True
                )[1]
                tf = time.time()
                time_probs = tf - t0
                    
                # Entropy-Complexity
                t0 = time.time()
                h, c = ordpy.complexity_entropy(data=data_probs, dx=embedding_dimension, probs=True)
                tf = time.time()
                time_hc.append(tf - t0 + time_probs)
                #Fisher-Shannon
                t0 = time.time()
                f, s = ordpy.fisher_shannon(data=data_probs, dx=embedding_dimension, probs=True)
                tf = time.time()
                time_fs.append(tf - t0 + time_probs)
                row[feature] = window[feature].values[-1]
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1

        new_df.to_csv(output_path, index=False)
        time_dict = {
            'time_hc': time_hc,
            'time_fs': time_fs
        }
        time_df = pd.DataFrame.from_dict(time_dict)
        time_df.to_csv(output_path_time, index=False)

        logger.info(f"Processamento concluído. Arquivo salvo em: {output_path} e {output_path_time}")
    else:
        logger.error("Arquivo de entrada não encontrado.")
    logger.info(f"Tempo total de processamento: {time.time() - exec_time:.2f} segundos")


def main():
    parser = argparse.ArgumentParser(
        description="Processa arquivos de entrada e saída."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Caminho do arquivo de entrada"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Diretório ou arquivo de saída"
    )

    parser.add_argument(
        "--window_length",
        type=int,
        required=True,
        help="Tamanho da janela de processamento"
    )

    parser.add_argument(
        "--embedding_dimension",
        type=int,
        required=True,
        help="Dimensão de incorporação"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Exemplo de leitura
    if input_path.exists():
        # Cria diretório de saída se necessário
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exemplo de escrita
        # with open(output_path, "w", encoding="utf-8") as f:
        #     f.write(content)

        process_data(
            input_path=input_path, 
            output_path=output_path, 
            window_length=args.window_length, 
            embedding_dimension=args.embedding_dimension
        )

        logger.info(f"Arquivo processado e salvo em: {output_path}")
    else:
        logger.error("Arquivo de entrada não encontrado.")


if __name__ == "__main__":
    main()