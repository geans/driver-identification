#!/bin/bash

# Script Bash para gerar a Tabela 2 em LaTeX a partir do arquivo JSON de métricas.
# Uso: ./generate_table.sh [caminho_do_arquivo_json]

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
        
    # Corrige os erros de sintaxe mais comuns no JSON (falta de vírgula entre as listas)
    content_fixed = content.strip()
    
    # 1. Caso os colchetes principais estejam colados ou com chaves sem vírgulas
    content_fixed = re.sub(r'\]\s*\[', '], [', content_fixed)
    content_fixed = re.sub(r'\}\s*\[', '}, [', content_fixed)
    content_fixed = re.sub(r'\]\s*\{', '], {', content_fixed)
    content_fixed = re.sub(r'\}\s*\{', '}, {', content_fixed)
    
    # Garantir que o JSON esteja envelopado em um array principal se necessário
    if not content_fixed.startswith('['):
        content_fixed = '[' + content_fixed + ']'
        
    try:
        data = json.loads(content_fixed)
    except Exception as e:
        # Tentativa de recuperação de emergência mais agressiva se falhar
        try:
            # Substitui o padrão específico do arquivo original: ]}[ por }], [{
            # Exemplo: ... 0.8833333333333333]}[{"kNN": ...
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
                    'std': compute_std(merged, mean),
                    'count': len(merged)
                }
        return stats

    stats_lit = get_stats(literatura)
    stats_prop = get_stats(proposta)
    
    # Imprime a tabela em formato LaTeX
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Comparação de Desempenho dos Classificadores: Literatura vs. Proposta (Dados Consolidados dos 4 Motoristas)}")
    print(r"  \label{tab:comparacao_classificadores}")
    print(r"  \begin{tabular}{llcccccc}")
    print(r"    \hline")
    print(r"    \textbf{Classificador} & \textbf{Experimento} & \textbf{Mínimo} & \textbf{Máximo} & \textbf{Média} & \textbf{Mediana} & \textbf{Desvio Padrão} & \textbf{N} \\")
    print(r"    \hline")
    
    for i, clf in enumerate(classifiers):
        lit = stats_lit.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0})
        prop = stats_prop.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'count': 0})
        
        def fmt(val):
            return f"{val:.4f}".replace('.', ',')
            
        print(f"    \\multirow{{2}}{{*}}{{{clf}}} & Literatura & {fmt(lit['min'])} & {fmt(lit['max'])} & {fmt(lit['mean'])} & {fmt(lit['median'])} & {fmt(lit['std'])} & {lit['count']} \\\\")
        print(f"                           & Proposta   & {fmt(prop['min'])} & {fmt(prop['max'])} & {fmt(prop['mean'])} & {fmt(prop['median'])} & {fmt(prop['std'])} & {prop['count']} \\\\")
        if i < len(classifiers) - 1:
            print(r"    \hline")
            
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    main()
EOF
