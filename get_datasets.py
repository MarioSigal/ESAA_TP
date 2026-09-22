"""
Script de descarga y preparación de datasets para el proyecto ESAA:
1. PTB-XL (Alemania - PhysioNet v1.0.3)
2. ICBEB 2018 / CPSC 2018 (China 12-Lead Challenge)

Permite dos modalidades:
  --mode fast : Descarga metadatos completos y señales de muestra para EDA instantáneo (< 30s).
  --mode full : Descarga los paquetes completos (~1.8 GB PTB-XL + señales ICBEB).
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil

PTB_XL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
PTB_XL_ZIP_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
ICBEB_REF_URL = "http://2018.icbeb.org/file/REFERENCE.csv"
ICBEB_VAL_URL = "http://2018.icbeb.org/file/validation_set.zip"
PHYSIONET_CPSC_BASE = "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"


def download_with_progress(url, dest_path, desc=""):
    class DownloadProgressBar:
        def __init__(self, desc):
            self.desc = desc
            self.last_percent = -1

        def update(self, block_num, block_size, total_size):
            if total_size <= 0:
                return
            percent = int(block_num * block_size * 100 / total_size)
            if percent != self.last_percent and percent <= 100:
                self.last_percent = percent
                sys.stdout.write(f"\r  [{self.desc}] {percent}% [{block_num * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]")
                sys.stdout.flush()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"  [Existente] {dest_path}")
        return

    print(f"  [Descargando] {url} -> {dest_path}")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, DownloadProgressBar(desc).update)
    print(" -> Completado.")


def setup_datasets(mode="fast", output_dir="data"):
    ptb_dir = os.path.join(output_dir, "ptbxl")
    icbeb_dir = os.path.join(output_dir, "ICBEB")
    os.makedirs(ptb_dir, exist_ok=True)
    os.makedirs(icbeb_dir, exist_ok=True)

    print("\n=======================================================")
    print(f"Iniciando descarga de datasets en modo: {mode.upper()}")
    print("=======================================================")

    # 1. Metadatos de PTB-XL
    print("\n1. Descargando metadatos de PTB-XL...")
    download_with_progress(PTB_XL_BASE_URL + "ptbxl_database.csv", os.path.join(ptb_dir, "ptbxl_database.csv"), "PTB-XL Database")
    download_with_progress(PTB_XL_BASE_URL + "scp_statements.csv", os.path.join(ptb_dir, "scp_statements.csv"), "SCP Statements")

    # 2. Metadatos de ICBEB
    print("\n2. Descargando metadatos de ICBEB 2018...")
    download_with_progress(ICBEB_REF_URL, os.path.join(icbeb_dir, "REFERENCE.csv"), "ICBEB Reference")
    shutil.copy(os.path.join(ptb_dir, "scp_statements.csv"), os.path.join(icbeb_dir, "scp_statements.csv"))

    if mode == "fast":
        print("\n3. Descargando señales de muestra representativas (Fast Mode)...")
        # PTB-XL samples
        ptb_samples = [
            ("records100/00000/00001_lr.hea", os.path.join(ptb_dir, "records100/00000/00001_lr.hea")),
            ("records100/00000/00001_lr.dat", os.path.join(ptb_dir, "records100/00000/00001_lr.dat")),
            ("records100/00000/00002_lr.hea", os.path.join(ptb_dir, "records100/00000/00002_lr.hea")),
            ("records100/00000/00002_lr.dat", os.path.join(ptb_dir, "records100/00000/00002_lr.dat")),
            ("records100/00000/00014_lr.hea", os.path.join(ptb_dir, "records100/00000/00014_lr.hea")),
            ("records100/00000/00014_lr.dat", os.path.join(ptb_dir, "records100/00000/00014_lr.dat")),
        ]
        for src, dst in ptb_samples:
            download_with_progress(PTB_XL_BASE_URL + src, dst, os.path.basename(dst))

        # ICBEB samples
        icbeb_samples = [
            ("g1/A0001.hea", os.path.join(icbeb_dir, "records500/A0001.hea")),
            ("g1/A0001.mat", os.path.join(icbeb_dir, "records500/A0001.mat")),
            ("g1/A0002.hea", os.path.join(icbeb_dir, "records500/A0002.hea")),
            ("g1/A0002.mat", os.path.join(icbeb_dir, "records500/A0002.mat")),
            ("g1/A0003.hea", os.path.join(icbeb_dir, "records500/A0003.hea")),
            ("g1/A0003.mat", os.path.join(icbeb_dir, "records500/A0003.mat")),
        ]
        for src, dst in icbeb_samples:
            download_with_progress(PHYSIONET_CPSC_BASE + src, dst, os.path.basename(dst))

        print("\n[OK] Modo Fast completado: Metadatos y señales de muestra listos.")

    elif mode == "full":
        print("\n3. Descargando archivo completo de PTB-XL (~1.8 GB)...")
        zip_dest = os.path.join(output_dir, "ptbxl_full.zip")
        download_with_progress(PTB_XL_ZIP_URL, zip_dest, "PTB-XL ZIP")
        print("  Descomprimiendo PTB-XL...")
        with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("  PTB-XL extraído exitosamente.")

        print("\n4. Descargando señales de validación de ICBEB (194 MB)...")
        val_dest = os.path.join(output_dir, "icbeb_val.zip")
        download_with_progress(ICBEB_VAL_URL, val_dest, "ICBEB Val ZIP")
        with zipfile.ZipFile(val_dest, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_dir, "tmp_icbeb"))
        print("  ICBEB extraído exitosamente.")

    print("\n=======================================================")
    print("Preparación de datos completada exitosamente.")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga de datasets PTB-XL e ICBEB")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Modo de descarga: fast (metadatos + muestras) o full (datasets completos)")
    parser.add_argument("--output_dir", default="data", help="Directorio destino")
    args = parser.parse_args()
    setup_datasets(mode=args.mode, output_dir=args.output_dir)
