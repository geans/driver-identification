import json
import math
import re
import sys


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
                max_data = max(merged)
                if max_data == 1.0:
                    max_data = 0.9999
                stats[clf] = {
                    'min': min(merged),
                    'max': max_data,
                    'mean': mean,
                    'median': compute_median(merged),
                    'std': compute_std(merged, mean)
                }
        return stats

    stats_lit = get_stats(literatura)
    stats_prop = get_stats(proposta)
    
    def fmt(val):
        return f"{val * 100:.2f}"

    # TABELA 1: LITERATURA
    print(r"% --- TABELA 1: LITERATURA ---")
    print(r"\begin{table*}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Performance of the Classifiers in the Literature}")
    print(r"  \label{tab:performance-literature}")
    print(r"  \begin{tabular}{lccccc}")
    print(r"    \hline")
    print(r"    \textbf{Classifier} & \textbf{Minimum} & \textbf{Maximum} & \textbf{Average} & \textbf{Median} & \textbf{Standard Deviation} \\")
    print(r"    \hline")
    for clf in classifiers:
        lit = stats_lit.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0})
        print(f"    {clf:<12} & {fmt(lit['min'])} & {fmt(lit['max'])} & {fmt(lit['mean'])} & {fmt(lit['median'])} & {fmt(lit['std'])} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table*}")
    print()

    # TABELA 2: PROPOSTA
    print(r"% --- TABELA 2: PROPOSTA ---")
    print(r"\begin{table*}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Performance of the Classifiers in the Proposal}")
    print(r"  \label{tab:performance-proposal}")
    print(r"  \begin{tabular}{lccccc}")
    print(r"    \hline")
    print(r"    \textbf{Classifier} & \textbf{Minimum} & \textbf{Maximum} & \textbf{Average} & \textbf{Median} & \textbf{Standard Deviation} \\")
    print(r"    \hline")
    for clf in classifiers:
        prop = stats_prop.get(clf, {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0})
        print(f"    {clf:<12} & {fmt(prop['min'])} & {fmt(prop['max'])} & {fmt(prop['mean'])} & {fmt(prop['median'])} & {fmt(prop['std'])} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table*}")

if __name__ == '__main__':
    main()