#!/bin/bash

###############################################################################
# Script para descargar y preparar los datasets PTB-XL e ICBEB 2018 (CPSC 2018)
# Actualizado con URLs activas de PhysioNet y mirrors estables
###############################################################################

set -e

echo "=== 1. DESCARGA DE METADATOS Y DATASET PTB-XL ==="
mkdir -p data/ptbxl
cd data/ptbxl

# URL oficial activa en PhysioNet (v1.0.3)
if [ ! -f "ptbxl_database.csv" ]; then
    echo "Descargando metadatos de PTB-XL..."
    wget -c https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
    wget -c https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
fi

# Opcional: descargar el archivo completo comprimido (~1.8 GB)
# Descomentar la siguiente línea para descargar las señales completas:
# wget -c https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
# unzip -q ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

cd ../..

echo "=== 2. DESCARGA DE METADATOS Y DATASET ICBEB (CPSC 2018) ==="
mkdir -p tmp_data
cd tmp_data

if [ ! -f "REFERENCE.csv" ]; then
    echo "Descargando REFERENCE.csv de ICBEB 2018..."
    wget -c http://2018.icbeb.org/file/REFERENCE.csv
fi

if [ ! -f "validation_set.zip" ]; then
    echo "Descargando validation_set.zip de ICBEB (194 MB)..."
    wget -c http://2018.icbeb.org/file/validation_set.zip
fi

cd ..

echo "=== Descarga inicial completada con éxito ==="
echo "Nota: Para la descarga completa multiplataforma con verificación, use: python get_datasets.py"
