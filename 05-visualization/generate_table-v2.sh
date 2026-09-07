#!/bin/bash

# Script Bash para gerar tabelas separadas em LaTeX a partir do arquivo JSON de métricas.
# Uso: ./generate_table-v2.sh [caminho_do_arquivo_json]

FILE="${1:-../04-fitting/results/experiment/analyse_information.out_values.txt}"

if [ ! -f "$FILE" ]; then
    echo "Erro: Arquivo '$FILE' não encontrado." >&2
    echo "Uso: $0 [caminho_do_arquivo_json]" >&2
    exit 1
fi

python3 - "$FILE" <<'EOF'
import json
import re
import sys
import math

def compute_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def compute_median(lst):
    if not lst:
        return 0.0
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 1:
        return sorted_lst[n // 2]
    else:
        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2.0

def compute_std(lst, mean):
    if len(lst) <= 1:
        return 0.0
    variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)

def main():
    filepath = sys.argv[1]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Erro ao ler o arquivo {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
        
    content_fixed = content.strip()
    
    content_fixed = re.sub(r'\]\s*\[', '], [', content_fixed)
    content_fixed = re.sub(r'\}\s*\[', '}, [', content_fixed)
    content_fixed = re.sub(r'\]\s*\{', '], {', content_fixed)
    content_fixed = re.sub(r'\}\s*\{', '}, {', content_fixed)
    
    if not content_fixed.startswith('['):
        content_fixed = '[' + content_fixed + ']'
        
    try:
        data = json.loads(content_fixed)
    except Exception as e:
        try:
            content_fixed = content.replace(']}[{"kNN"', ']}], [{"kNN"')
            if not content_fixed.startswith('['):
                content_fixed = '[' + content_fixed + ']'
            data = json.loads(content_fixed)
        except Exception as e2:
            print(f"Erro ao decodificar JSON: {e2}", file=sys.stderr)
            sys.exit(1)
            
    if len(data) < 2:
        print("Erro: O JSON deve conter pelo menos duas listas principais (Literatura e Proposta).", file=sys.stderr)
        sys.exit(1)
        
    literatura = data[0]
    proposta = data[1]
    
    classifiers = ["kNN", "Linear SVM", "RBF SVM", "D. Tree", "R. Forest", "MLP", "N. Bayes"]
    
    def get_stats(dataset):
        stats = {}
        for clf in classifiers:
            merged = []
            for driver in dataset:
                if clf in driver:
                    merged.extend(driver[clf])
            if merged:
                mean = compute_mean(merged)
                stats[clf] = {
                    'min': min(merged),
                    'max': max(merged),
                    'mean': mean,
                    'median': compute_median(merged),
                    'std': compute_std(merged, mean)
                }
        return stats

    stats_lit = get_stats(literatura)
    stats_prop = get_stats(proposta)
    
    def fmt(val):
        return f"{val:.4f}".replace('.', ',')

    # TABELA 1: LITERATURA
    print(r"% --- TABELA 1: LITERATURA ---")
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Métricas de Desempenho dos Classificadores na Literatura (Dados Consolidados dos 4 Motoristas)}")
    print(r"  \label{tab:desempenho_literatura}")
    print(r"  \begin{tabular}{lccccc}")
    print(r"    \hline")
    print(r"    \textbf{Classificador} & \textbf{Mínimo} & \textbf{Máximo} & \textbf{Média} & \textbf{Mediana} & \textbf{Desvio Padrão} \\")
    print(r"    \hline")
    for clf in classifiers:
        lit = stats_lit.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0})
        print(f"    {clf:<12} & {fmt(lit['min'])} & {fmt(lit['max'])} & {fmt(lit['mean'])} & {fmt(lit['median'])} & {fmt(lit['std'])} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()

    # TABELA 2: PROPOSTA
    print(r"% --- TABELA 2: PROPOSTA ---")
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Métricas de Desempenho dos Classificadores na Proposta (Dados Consolidados dos 4 Motoristas)}")
    print(r"  \label{tab:desempenho_proposta}")
    print(r"  \begin{tabular}{lccccc}")
    print(r"    \hline")
    print(r"    \textbf{Classificador} & \textbf{Mínimo} & \textbf{Máximo} & \textbf{Média} & \textbf{Mediana} & \textbf{Desvio Padrão} \\")
    print(r"    \hline")
    for clf in classifiers:
        prop = stats_prop.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0})
        print(f"    {clf:<12} & {fmt(prop['min'])} & {fmt(prop['max'])} & {fmt(prop['mean'])} & {fmt(prop['median'])} & {fmt(prop['std'])} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    main()
EOF
