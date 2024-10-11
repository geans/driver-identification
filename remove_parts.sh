#!/bin/bash

# Diretório principal, passe como argumento ou defina diretamente
main_dir="$1"

# Verifica se o diretório principal foi fornecido
if [ -z "$main_dir" ]; then
    echo "Uso: $0 <diretório_principal>"
    exit 1
fi

# Verifica se o diretório existe
if [ ! -d "$main_dir" ]; then
    echo "Diretório não encontrado: $main_dir"
    exit 1
fi

# Verifica se o diretório existe
trashed="$main_dir"/trashed
if [ ! -d "$trashed" ]; then
    mkdir "$trashed"
fi

# Loop em todas as subpastas do diretório principal
for dir in "$main_dir"/*; do
    # Verifica se é um diretório
    if [ -d "$dir" ]; then
        echo "Entrando no diretório: $dir"
        
        # Loop em cada subdiretório A, B, C, D
        for subdir in "$dir"/{A,B,C,D}; do
            if [ -d "$subdir" ]; then
                echo "Removendo arquivos descartáveis do diretório: $subdir"
                # Remove os arquivos que seguem o padrão All_*.csv.part*.csv*
                mv "$subdir"/All_*.csv.part*.csv* "$trashed"
                
                # Exibe uma mensagem de sucesso ou se não encontrou arquivos
                if [ $? -eq 0 ]; then
                    echo "Arquivos removidos em: $subdir"
                else
                    echo "Nenhum arquivo correspondente encontrado em: $subdir"
                fi
            else
                echo "Subdiretório não encontrado: $subdir"
            fi
        done
    fi
done

echo "Os arquivos encontrados foram movidos para $trashed."