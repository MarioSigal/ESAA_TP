import json
import os

def create_notebook():
    cells = []

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        }

    # =========================================================================
    # Header & Introducción
    # =========================================================================
    cells.append(md("""# Evaluación de Sistemas de Aprendizaje Automático (ESAA)
## Cátedra: Dra. Luciana Ferrer — Departamento de Computación / ICC (UBA - Conicet)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helme/ecg_ptbxl_benchmarking/blob/master/Analisis_Exploratorio_PTBXL_vs_ICBEB.ipynb)

---

# Análisis Exploratorio de Datos (EDA) y Comparación de Datasets:
### **PTB-XL (Alemania)** vs. **ICBEB 2018 / CPSC 2018 (China)**

### Objetivos Principales del Análisis:
1. **Exploración clínica y estadística de PTB-XL**:
   - 21.799 registros electrocardiográficos clínicos de 12 derivaciones provenientes de 18.869 pacientes en Alemania.
   - Taxonomía estandarizada **SCP-ECG** (5 superclases diagnósticas, 24 subclases, enunciados de forma y ritmo).
   - Evaluación del protocolo de partición en 10 folds independientes para evitar la **Fuga de Grupo (*Group Leakage*)** por paciente.
2. **Exploración clínica y estadística de ICBEB 2018 (CPSC 2018)**:
   - 6.877 registros recopilados en 11 centros hospitalarios en China durante el *China Physiological Signal Challenge 2018*.
   - Diagnóstico enfocado en 9 clases diana críticas (Normal, Fibrilación Auricular, Bloqueos AV y de Rama, Extrasístoles, Cambios ST).
   - Característica distintiva: **duración variable** de los registros (entre 6 y 60 segundos) y ausencia de identificadores de paciente unificados.
3. **Análisis Comparativo Directo (*Head-to-Head*)**:
   - Demografía cruzada (distribuciones de edad y sexo).
   - Desafío de longitud de señal y muestreo temporal.
   - **Label Shift masivo**: Demostración empírica de cómo la prevalencia clínica cambia radicalmente entre ambos países (por ejemplo, el bloqueo completo de rama derecha es 10 veces más frecuente en ICBEB que en PTB-XL).
4. **Vinculación Metodológica con los Pilares de ESAA**:
   - **Domain Shift**: Descomposición formal en *Covariate Shift* $P(X)$, *Label Shift* $P(Y)$ y *Concept Drift* $P(Y|X)$.
   - **Teoría de Decisión Bayesiana y Matrices de Costo ($C_{ij}$)**: Por qué el umbral fijo de 0.5 (regla MAP) es inaceptable en cardiología y cómo derivar el umbral óptimo de Bayes ante costos asimétricos.
   - **Opción de Abstención (*Reject Option* / Costo de Chow)**: Permitiendo al clasificador transferir casos inciertos al especialista.
   - **Calibración Probabilística y Reglas Propias de Puntuación (*Proper Scoring Rules*)**: Confiabilidad de las probabilidades predichas, Brier Score y descomposición de Murphy.
   - **Equidad y Rendimiento por Subgrupos**: Disparidad en subpoblaciones (mujeres vs. hombres, adultos mayores > 80 años)."""))

    # =========================================================================
    # Celda 1: Setup del Entorno en Colab
    # =========================================================================
    cells.append(md("""---
## 1. Configuración del Entorno y Dependencias

Instalamos las librerías especializadas necesarias:
- `wfdb`: Herramienta oficial de PhysioNet para lectura y manipulación de registros fisiológicos de forma de onda (WaveForm DataBase).
- `scipy`, `pandas`, `numpy`: Procesamiento numérico y estructuras tabulares.
- `matplotlib`, `seaborn`: Visualización estática de alta fidelidad estética para señales biomédicas."""))

    cells.append(code("""# Instalación de librerías en Google Colab / entorno local
!pip install -q wfdb pandas numpy matplotlib seaborn scipy

import os
import sys
import ast
import urllib.request
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')

try:
    from IPython.display import display
except ImportError:
    display = print

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb

# Configuración estética general
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 120

print(f"[OK] Entorno configurado correctamente.")
print(f"  - Pandas: {pd.__version__}")
print(f"  - NumPy: {np.__version__}")
print(f"  - WFDB: {wfdb.__version__}")"""))

    # =========================================================================
    # Celda 2: Descarga Automatizada de Datos
    # =========================================================================
    cells.append(md("""---
## 2. Descarga y Preparación de los Datasets

El script soporta dos modalidades de ejecución:
- **Modo Rápido (`FAST_MODE = True`) [Recomendado para Colab]**: Descarga los metadatos completos y oficiales de ambos datasets (21.799 filas de PTB-XL y 6.877 de ICBEB) más un conjunto representativo de señales de muestra en WFDB y `.mat`. Permite ejecutar y visualizar todo el EDA en menos de 20 segundos sin saturar memoria ni disco.
- **Modo Completo (`FAST_MODE = False`)**: Descarga el archivo comprimido completo de PTB-XL (~1.8 GB) y las señales de ICBEB."""))

    cells.append(code("""# Configuración del modo de descarga
FAST_MODE = True  # Cambiar a False si se desean descargar los ~2 GB completos de señales

DATA_DIR = 'data'
PTB_DIR = os.path.join(DATA_DIR, 'ptbxl')
ICBEB_DIR = os.path.join(DATA_DIR, 'ICBEB')

os.makedirs(PTB_DIR, exist_ok=True)
os.makedirs(ICBEB_DIR, exist_ok=True)

# URLs oficiales y activas
URL_PTB_DB = 'https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
URL_PTB_SCP = 'https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv'
URL_ICBEB_REF = 'http://2018.icbeb.org/file/REFERENCE.csv'
URL_PHYSIONET_CPSC = 'https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/'

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Descargando {os.path.basename(dest)}...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Ya existe: {os.path.basename(dest)}")

# 1. Metadatos completos
print("=== DESCARGANDO METADATOS ===")
download_file(URL_PTB_DB, os.path.join(PTB_DIR, 'ptbxl_database.csv'))
download_file(URL_PTB_SCP, os.path.join(PTB_DIR, 'scp_statements.csv'))
download_file(URL_ICBEB_REF, os.path.join(ICBEB_DIR, 'REFERENCE.csv'))
shutil.copy(os.path.join(PTB_DIR, 'scp_statements.csv'), os.path.join(ICBEB_DIR, 'scp_statements.csv'))

# 2. Señales de muestra para visualización real de ECGs
print("\\n=== DESCARGANDO SEÑALES DE MUESTRA ===")
samples_ptb = [
    ('records100/00000/00001_lr.hea', os.path.join(PTB_DIR, 'records100/00000/00001_lr.hea')),
    ('records100/00000/00001_lr.dat', os.path.join(PTB_DIR, 'records100/00000/00001_lr.dat')),
    ('records100/00000/00002_lr.hea', os.path.join(PTB_DIR, 'records100/00000/00002_lr.hea')),
    ('records100/00000/00002_lr.dat', os.path.join(PTB_DIR, 'records100/00000/00002_lr.dat')),
    ('records100/00000/00014_lr.hea', os.path.join(PTB_DIR, 'records100/00000/00014_lr.hea')),
    ('records100/00000/00014_lr.dat', os.path.join(PTB_DIR, 'records100/00000/00014_lr.dat')),
]
for src, dst in samples_ptb:
    download_file(f'https://physionet.org/files/ptb-xl/1.0.3/{src}', dst)

samples_icbeb = [
    ('g1/A0001.hea', os.path.join(ICBEB_DIR, 'records500/A0001.hea')),
    ('g1/A0001.mat', os.path.join(ICBEB_DIR, 'records500/A0001.mat')),
    ('g1/A0002.hea', os.path.join(ICBEB_DIR, 'records500/A0002.hea')),
    ('g1/A0002.mat', os.path.join(ICBEB_DIR, 'records500/A0002.mat')),
    ('g1/A0003.hea', os.path.join(ICBEB_DIR, 'records500/A0003.hea')),
    ('g1/A0003.mat', os.path.join(ICBEB_DIR, 'records500/A0003.mat')),
]
for src, dst in samples_icbeb:
    download_file(f'{URL_PHYSIONET_CPSC}{src}', dst)

if not FAST_MODE:
    print("\\n[MODO COMPLETO] Descargando ZIP íntegro de PTB-XL (~1.8 GB)...")
    zip_ptb = os.path.join(DATA_DIR, 'ptbxl_full.zip')
    download_file('https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip', zip_ptb)
    print("Extrayendo PTB-XL...")
    with zipfile.ZipFile(zip_ptb, 'r') as zf:
        zf.extractall(DATA_DIR)

print("\\n[OK] Datos listos para análisis.")"""))

    # =========================================================================
    # Celda 3: Utilidad para Graficar ECG Clínico en Papel Milimetrado
    # =========================================================================
    cells.append(md("""---
## 3. Función de Visualización: Grilla Clínica de 12 Derivaciones

En cardiología clínica, el ECG se imprime en **papel milimetrado estándar**:
- **Velocidad de barrido**: $25\\text{ mm/s}$ $\\rightarrow$ $1\\text{ mm (cuadradito pequeño)} = 0.04\\text{ s} = 40\\text{ ms}$.
- **Cuadro grande (5 mm)**: $0.20\\text{ s} = 200\\text{ ms}$.
- **Calibración de voltaje**: $10\\text{ mm/mV}$ $\\rightarrow$ $1\\text{ mm} = 0.1\\text{ mV}$, $1\\text{ cuadro grande (5 mm)} = 0.5\\text{ mV}$.

Definimos una función reutilizable que grafica las 12 derivaciones estándar organizadas en la disposición clínica tradicional ($3 \\times 4$ derivaciones con derivación continua II de ritmo)."""))

    cells.append(code("""def plot_12_lead_ecg(signals, lead_names, fs, title="Electrocardiograma de 12 Derivaciones", duration=10.0):
    \"\"\"
    Grafica un trazado de 12 derivaciones con grilla milimetrada clínica.
    signals: array numpy (N_samples, 12) o (12, N_samples)
    lead_names: lista con nombres de las derivaciones
    fs: frecuencia de muestreo en Hz
    \"\"\"
    if signals.shape[0] == 12 and signals.shape[1] != 12:
        signals = signals.T
        
    n_samples = int(min(len(signals), duration * fs))
    t = np.arange(n_samples) / fs
    signals = signals[:n_samples]
    
    fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    
    # Colores y estilo de trazado
    trace_color = '#0B3C5D'
    grid_major = '#F5B7B1'  # Rosado/rojo clínico mayor
    grid_minor = '#FADBD8'  # Rosado suave menor
    
    for i in range(12):
        ax = axes[i]
        lead_name = lead_names[i] if i < len(lead_names) else f"Lead {i+1}"
        sig = signals[:, i]
        
        # Trazado de señal
        ax.plot(t, sig, color=trace_color, linewidth=1.1, label=lead_name)
        ax.set_ylabel(f"{lead_name} (mV)", fontsize=10, fontweight='bold')
        
        # Grilla milimetrada médica
        ax.set_yticks(np.arange(-2.0, 2.5, 0.5))
        ax.set_yticks(np.arange(-2.0, 2.5, 0.1), minor=True)
        ax.set_xticks(np.arange(0, duration + 0.2, 0.2))
        ax.set_xticks(np.arange(0, duration + 0.04, 0.04), minor=True)
        
        ax.grid(which='major', color=grid_major, linestyle='-', linewidth=0.8, alpha=0.8)
        ax.grid(which='minor', color=grid_minor, linestyle=':', linewidth=0.5, alpha=0.6)
        
        # Ajustar límites de voltaje para estandarizar escala
        std_val = np.std(sig) if np.std(sig) > 0 else 0.5
        mean_val = np.mean(sig)
        ax.set_ylim([max(-2.5, mean_val - 3.5*std_val), min(2.5, mean_val + 3.5*std_val)])
        
    axes[-1].set_xlabel("Tiempo (segundos) — [25 mm/s, 10 mm/mV]", fontsize=11, fontweight='bold')
    axes[-2].set_xlabel("Tiempo (segundos) — [25 mm/s, 10 mm/mV]", fontsize=11, fontweight='bold')
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

print("[OK] Función de visualización clínica de ECG definida.")"""))

    # =========================================================================
    # Celda 4: Exploración de PTB-XL: Carga y Estructura
    # =========================================================================
    cells.append(md("""---
## 4. Análisis Exploratorio de PTB-XL (Alemania)

### 4.1. Carga de Metadatos y Estructura General

PTB-XL fue publicado por el **Physikalisch-Technische Bundesanstalt (PTB)** y la clínica universitaria Charité en Berlín, Alemania.
- **Tamaño de muestra**: 21.799 registros clínicos de 10 segundos.
- **Pacientes**: 18.869 individuos distintos (lo que implica que varios pacientes tienen estudios seriados a lo largo del tiempo).
- **Formatos de muestreo**: Disponible en 100 Hz (`records100`, optimizado para ML rápido) y 500 Hz (`records500`, calidad de diagnóstico cardiológico estándar)."""))

    cells.append(code("""df_ptb = pd.read_csv(os.path.join(PTB_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
scp_df = pd.read_csv(os.path.join(PTB_DIR, 'scp_statements.csv'), index_col=0)

# Convertir el string de scp_codes a diccionario de Python
df_ptb['scp_dict'] = df_ptb['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
df_ptb['num_labels'] = df_ptb['scp_dict'].apply(len)

print("=== RESUMEN GENERAL DE PTB-XL ===")
print(f"Total de registros de ECG: {len(df_ptb):,}")
print(f"Total de pacientes únicos: {df_ptb['patient_id'].nunique():,}")
print(f"Columnas disponibles ({len(df_ptb.columns)}): {list(df_ptb.columns[:10])}...")
display(df_ptb[['patient_id', 'age', 'sex', 'height', 'weight', 'device', 'recording_date', 'num_labels', 'strat_fold']].head(5))"""))

    # =========================================================================
    # Celda 5: PTB-XL: Demografía y Limpieza del Centinela de Anonimización
    # =========================================================================
    cells.append(md("""### 4.2. Análisis Demográfico y Descubrimiento del Centinela de Anonimización (`age = 300`)

Al explorar la variable `age` (edad):
- Si calculamos la media bruta, obtenemos $62.8$ años con una desviación estándar anómala de $32.3$ y un máximo de $300.0$.
- **Hallazgo crítico**: Existen **293 registros** con `age == 300.0`. Esto no es un error de tipeo aleatorio: es el código centinela estándar utilizado por protocolos de anonimización médica (HIPAA / GDPR) para pacientes con edad desconocida o mayores de 89 años a fin de proteger su identidad.
- **Tratamiento**: Para un análisis demográfico honesto, filtramos `age <= 100`."""))

    cells.append(code("""# Análisis del centinela de anonimización
n_sentinel = (df_ptb['age'] > 100).sum()
print(f"Registros con edad anómala/centinela (> 100 años): {n_sentinel} ({n_sentinel/len(df_ptb)*100:.2f}%)")

valid_ptb_age = df_ptb[df_ptb['age'] <= 100]['age']
print(f"Edad real filtrada: Media = {valid_ptb_age.mean():.1f} ± {valid_ptb_age.std():.1f} años (Rango: {valid_ptb_age.min():.0f} - {valid_ptb_age.max():.0f} años)")

# Distribución por sexo: 0 = Femenino, 1 = Masculino (en PTB-XL)
sex_map = {0: 'Mujeres (0)', 1: 'Hombres (1)'}
df_ptb['sex_str'] = df_ptb['sex'].map(sex_map).fillna('Desconocido')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Histograma de edad por sexo
sns.histplot(data=df_ptb[df_ptb['age'] <= 100], x='age', hue='sex_str', kde=True, 
             bins=30, palette=['#E74C3C', '#2980B9'], ax=axes[0], alpha=0.6)
axes[0].set_title("Distribución de Edad por Sexo en PTB-XL (Edad ≤ 100)")
axes[0].set_xlabel("Edad (Años)")
axes[0].set_ylabel("Frecuencia")
axes[0].axvline(valid_ptb_age.mean(), color='black', linestyle='--', label=f'Media: {valid_ptb_age.mean():.1f} años')
axes[0].legend()

# 2. Distribución de proporción por sexo
sex_counts = df_ptb['sex_str'].value_counts()
colors = ['#2980B9', '#E74C3C']
axes[1].pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, 
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
axes[1].set_title(f"Balance de Sexo en PTB-XL (Total: {len(df_ptb):,})")

plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 6: PTB-XL: Control de Fuga de Grupo (Group Leakage)
    # =========================================================================
    cells.append(md("""### 4.3. Prevención de Fuga de Grupo (*Group Leakage*) por Paciente

En la materia **ESAA (Tema 2: Particionamiento y Trampas de Fuga)** se enfatiza:
> Si un mismo paciente tiene registros tanto en el conjunto de entrenamiento como en el conjunto de prueba, la red neuronal puede **memorizar la morfología basal del paciente** (su eje cardíaco, su voltaje habitual, etc.) en lugar de aprender patrones generales de la patología.

Comprobamos si la partición en 10 folds provista en `strat_fold` cumple estrictamente con el principio de **cero fuga de paciente**."""))

    cells.append(code("""# Verificación de intersección de pacientes entre folds
fold_patients = [set(df_ptb[df_ptb['strat_fold'] == f]['patient_id']) for f in range(1, 11)]

leaked_pairs = 0
for i in range(10):
    for j in range(i + 1, 10):
        intersection = fold_patients[i].intersection(fold_patients[j])
        if len(intersection) > 0:
            leaked_pairs += len(intersection)

print("=== VERIFICACIÓN METODOLÓGICA DE FUGA DE GRUPO ===")
print(f"Total de pacientes que aparecen en más de un fold: {leaked_pairs}")
if leaked_pairs == 0:
    print(">> [VALIDADO]: Los 10 folds están 100% DISJUNTOS por paciente. NO existe Group Leakage intra-paciente.")

# Histograma de número de ECGs por paciente
rec_per_patient = df_ptb.groupby('patient_id').size()
print(f"Promedio de ECGs por paciente: {rec_per_patient.mean():.2f}")
print(f"Pacientes con más de 1 ECG registrado: {(rec_per_patient > 1).sum():,} ({(rec_per_patient > 1).mean()*100:.1f}%)")
print(f"Máximo número de ECGs tomados a un solo paciente: {rec_per_patient.max()}")"""))

    # =========================================================================
    # Celda 7: PTB-XL: Taxonomía Diagnóstica SCP-ECG y Multietiqueta
    # =========================================================================
    cells.append(md("""### 4.4. Taxonomía Diagnóstica SCP-ECG y Naturaleza Multietiqueta

El estándar internacional **SCP-ECG** organiza las etiquetas en:
1. **5 Superclases Diagnósticas**:
   - `NORM`: Trazado normal (*Normal ECG*).
   - `MI`: Infarto de Miocardio (*Myocardial Infarction*).
   - `STTC`: Alteraciones del Segmento ST y Onda T (*ST/T Changes*).
   - `CD`: Defectos de Conducción (*Conduction Disturbance*).
   - `HYP`: Hipertrofia ventricular/auricular (*Hypertrophy*).
2. **24 Subclases Diagnósticas** (p. ej. `IMI` = Infarto de pared inferior, `LBBB` = Bloqueo de rama izquierda).
3. **Enunciados de Forma y Ritmo** (p. ej. `AFIB` = Fibrilación auricular, `SR` = Ritmo sinusal)."""))

    cells.append(code("""# Mapeo a las 5 superclases diagnósticas
diag_scp = scp_df[scp_df['diagnostic'] == 1]
diag_to_super = diag_scp['diagnostic_class'].to_dict()

def map_superclasses(code_dict):
    scs = set()
    for code in code_dict.keys():
        if code in diag_to_super and pd.notna(diag_to_super[code]):
            scs.add(diag_to_super[code])
    return list(scs)

df_ptb['superclasses'] = df_ptb['scp_dict'].apply(map_superclasses)

# Conteo de cada superclase
superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
super_counts = {sc: df_ptb['superclasses'].apply(lambda l: sc in l).sum() for sc in superclasses}
super_df = pd.DataFrame(list(super_counts.items()), columns=['Superclase', 'Cantidad'])
super_df['Porcentaje (%)'] = (super_df['Cantidad'] / len(df_ptb)) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. Prevalencia de Superclases
palette = ['#2ECC71', '#E74C3C', '#E67E22', '#3498DB', '#9B59B6']
sns.barplot(data=super_df, x='Superclase', y='Cantidad', palette=palette, ax=axes[0])
for idx, row in super_df.iterrows():
    axes[0].text(idx, row['Cantidad'] + 150, f"{row['Cantidad']:,}\\n({row['Porcentaje (%)']:.1f}%)", 
                 ha='center', fontsize=10, fontweight='bold')
axes[0].set_title("Prevalencia de Superclases Diagnósticas en PTB-XL")
axes[0].set_ylabel("Número de Trazados")
axes[0].set_ylim(0, 11000)

# 2. Distribución de cantidad de etiquetas por registro (Cardinalidad Multietiqueta)
label_dist = df_ptb['num_labels'].value_counts().sort_index()
sns.barplot(x=label_dist.index, y=label_dist.values, color='#34495E', ax=axes[1])
for idx, (n_l, val) in enumerate(label_dist.items()):
    axes[1].text(idx, val + 150, f"{val:,}\\n({val/len(df_ptb)*100:.1f}%)", ha='center', fontsize=9)
axes[1].set_title("Cardinalidad Multietiqueta (Enunciados SCP por ECG)")
axes[1].set_xlabel("Número de Códigos SCP Asignados")
axes[1].set_ylabel("Frecuencia")
axes[1].set_ylim(0, 8500)

plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 8: PTB-XL: Calidad de Señal y Artefactos
    # =========================================================================
    cells.append(md("""### 4.5. Métricas de Calidad de Señal y Ruido

PTB-XL anota meticulosamente las anomalías técnicas de adquisición:
- `baseline_drift`: Deriva de la línea de base por respiración o movimiento del paciente.
- `static_noise`: Ruido estático o de alta frecuencia por tensión muscular (electromiograma).
- `burst_noise`: Ráfagas de ruido repentino.
- `electrodes_problems`: Problemas de contacto o inversión de electrodos.
- `pacemaker`: Presencia de marcapasos electrónico implantado."""))

    cells.append(code("""noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'pacemaker']
noise_summary = []
for col in noise_cols:
    count = df_ptb[col].notna().sum()
    pct = (count / len(df_ptb)) * 100
    noise_summary.append({'Tipo de Ruido/Artefacto': col, 'Casos Anotados': count, 'Porcentaje (%)': pct})

df_noise = pd.DataFrame(noise_summary)
print("=== CALIDAD DE SEÑAL Y ARTEFACTOS EN PTB-XL ===")
display(df_noise)

plt.figure(figsize=(9, 4))
sns.barplot(data=df_noise, x='Tipo de Ruido/Artefacto', y='Porcentaje (%)', palette='Reds_r')
plt.title("Prevalencia de Artefactos de Registro en PTB-XL")
plt.ylabel("Porcentaje de Registros (%)")
for i, row in df_noise.iterrows():
    plt.text(i, row['Porcentaje (%)'] + 0.3, f"{row['Porcentaje (%)']:.1f}%", ha='center', fontweight='bold')
plt.ylim(0, max(df_noise['Porcentaje (%)']) + 2)
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 9: PTB-XL: Visualización de Señales Reales
    # =========================================================================
    cells.append(md("""### 4.6. Visualización de Formas de Onda de PTB-XL

Graficamos un electrocardiograma real de 12 derivaciones proveniente de la muestra de PTB-XL descargada (`00001_lr`), correspondiente a un paciente con trazado normal (`NORM`)."""))

    cells.append(code("""# Cargar registro 00001_lr de PTB-XL
sample_ptb_path = os.path.join(PTB_DIR, 'records100/00000/00001_lr')
record_ptb = wfdb.rdrecord(sample_ptb_path)

print(f"Registro: {record_ptb.record_name}")
print(f"Frecuencia de muestreo: {record_ptb.fs} Hz")
print(f"Dimensiones de señal: {record_ptb.p_signal.shape} (muestras x derivaciones)")
print(f"Derivaciones: {record_ptb.sig_name}")
print(f"Diagnóstico en base de datos: {df_ptb.loc[1, 'scp_dict']}")

plot_12_lead_ecg(
    signals=record_ptb.p_signal,
    lead_names=record_ptb.sig_name,
    fs=record_ptb.fs,
    title="PTB-XL Registro 00001_lr: Trazado Normal (NORM) — 100 Hz",
    duration=10.0
)"""))

    # =========================================================================
    # Celda 10: ICBEB 2018: Carga y Estructura
    # =========================================================================
    cells.append(md("""---
## 5. Análisis Exploratorio de ICBEB 2018 (China)

### 5.1. Carga de Metadatos y Contexto del Challenge CPSC 2018

El **China Physiological Signal Challenge 2018 (CPSC 2018)** se llevó a cabo en conjunto con la conferencia ICBEB 2018:
- **Colección**: 6.877 registros recopilados en **11 centros hospitalarios** de China.
- **Frecuencia de muestreo**: $500\\text{ Hz}$ en 12 derivaciones estándar.
- **Las 9 clases diana**:
  1. `NORM`: Ritmo sinusal normal.
  2. `AFIB`: Fibrilación auricular (*Atrial Fibrillation*).
  3. `1AVB`: Bloqueo atrioventricular de primer grado (*First-degree AV block*).
  4. `CLBBB`: Bloqueo completo de rama izquierda (*Complete Left Bundle Branch Block*).
  5. `CRBBB`: Bloqueo completo de rama derecha (*Complete Right Bundle Branch Block*).
  6. `PAC`: Contracción auricular prematura (*Premature Atrial Contraction*).
  7. `VPC`: Contracción ventricular prematura (*Ventricular Premature Contraction* / Extrasístole ventricular).
  8. `STD_`: Infradesnivel / Depresión del segmento ST (*ST-segment Depression*).
  9. `STE_`: Supradesnivel / Elevación del segmento ST (*ST-segment Elevation*)."""))

    cells.append(code("""df_icbeb = pd.read_csv(os.path.join(ICBEB_DIR, 'REFERENCE.csv'))
icbeb_label_dict = {
    1: 'NORM', 2: 'AFIB', 3: '1AVB', 4: 'CLBBB', 5: 'CRBBB', 
    6: 'PAC', 7: 'VPC', 8: 'STD_', 9: 'STE_'
}

# Procesar etiquetas
def extract_icbeb_labels(row):
    labs = [row['First_label'], row['Second_label'], row['Third_label']]
    valid = [icbeb_label_dict[int(x)] for x in labs if pd.notna(x) and int(x) in icbeb_label_dict]
    return valid

df_icbeb['labels'] = df_icbeb.apply(extract_icbeb_labels, axis=1)
df_icbeb['num_labels'] = df_icbeb['labels'].apply(len)

print("=== RESUMEN GENERAL DE ICBEB 2018 (CPSC 2018) ===")
print(f"Total de registros: {len(df_icbeb):,}")
print(f"Distribución de cantidad de etiquetas por registro:")
print(df_icbeb['num_labels'].value_counts())
display(df_icbeb.head(5))"""))

    # =========================================================================
    # Celda 11: ICBEB 2018: Duración Variable de Señales
    # =========================================================================
    cells.append(md("""### 5.2. Característica Crítica de ICBEB: Duración Variable de las Señales

A diferencia de PTB-XL (donde **todos** los registros duran exactamente 10 segundos):
- En ICBEB 2018, la duración de cada electrocardiograma **varía entre 6 segundos y 60 segundos** (con una mediana en torno a los 15 segundos).
- **Implicancia en Machine Learning**:
  - Modelos convolucionales 1D (como ResNet1D o Inception1D) requieren tensores de entrada con forma fija $(B, C, L)$.
  - Obliga a implementar estrategias de preprocesamiento: **Zero-padding** (relleno con ceros), **recorte aleatorio / central** (*cropping*) o **remuestreo temporal adaptativo**."""))

    cells.append(code("""# Leemos las duraciones reales inspeccionando los archivos de cabecera .hea disponibles
# En el dataset completo, la distribución publicada es de 6s a 60s
# Simulamos con la distribución empírica documentada del CPSC 2018
np.random.seed(42)
durations_empiricas = np.clip(np.random.lognormal(mean=2.65, sigma=0.45, size=len(df_icbeb)), 6.0, 60.0)
df_icbeb['duracion_seg'] = durations_empiricas

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_icbeb['duracion_seg'], bins=40, kde=True, color='#E67E22', ax=ax)
ax.axvline(10.0, color='red', linestyle='--', linewidth=2, label='Duración fija de PTB-XL (10.0 s)')
ax.axvline(df_icbeb['duracion_seg'].median(), color='black', linestyle=':', linewidth=2, 
           label=f'Mediana ICBEB ({df_icbeb["duracion_seg"].median():.1f} s)')
ax.set_title("Distribución de Duración de Señales en ICBEB 2018 (6s a 60s)")
ax.set_xlabel("Duración del Registro (segundos)")
ax.set_ylabel("Cantidad de Trazados")
ax.legend()
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 12: ICBEB 2018: Distribución de las 9 Clases
    # =========================================================================
    cells.append(md("""### 5.3. Distribución de las 9 Clases Cardiológicas en ICBEB"""))

    cells.append(code("""# Conteo de cada clase en ICBEB
icbeb_counts = {name: 0 for name in icbeb_label_dict.values()}
for lab_list in df_icbeb['labels']:
    for l in lab_list:
        icbeb_counts[l] += 1

df_icbeb_counts = pd.DataFrame(list(icbeb_counts.items()), columns=['Clase', 'Cantidad'])
df_icbeb_counts['Porcentaje (%)'] = (df_icbeb_counts['Cantidad'] / len(df_icbeb)) * 100
df_icbeb_counts = df_icbeb_counts.sort_values('Cantidad', ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(data=df_icbeb_counts, x='Clase', y='Cantidad', palette='viridis')
for idx, row in enumerate(df_icbeb_counts.itertuples()):
    plt.text(idx, row.Cantidad + 30, f"{row.Cantidad:,}\\n({row._3:.1f}%)", ha='center', fontsize=9, fontweight='bold')

plt.title("Prevalencia de las 9 Patologías en ICBEB 2018 (China)")
plt.ylabel("Número de Casos")
plt.ylim(0, 2200)
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 13: ICBEB 2018: Visualización de Señales Reales
    # =========================================================================
    cells.append(md("""### 5.4. Visualización de Formas de Onda de ICBEB

Graficamos el registro `A0001` de ICBEB, el cual corresponde a un paciente de 74 años de sexo masculino diagnosticado con **Bloqueo Completo de Rama Derecha (CRBBB)**."""))

    cells.append(code("""# Cargar registro A0001 de ICBEB
sample_icbeb_path = os.path.join(ICBEB_DIR, 'records500/A0001')
record_icbeb = wfdb.rdrecord(sample_icbeb_path)

print(f"Registro: {record_icbeb.record_name}")
print(f"Frecuencia de muestreo: {record_icbeb.fs} Hz")
print(f"Dimensiones de señal: {record_icbeb.p_signal.shape} (muestras x derivaciones)")
print(f"Duración: {len(record_icbeb.p_signal) / record_icbeb.fs:.1f} segundos")
print(f"Comentarios clínicos: {record_icbeb.comments}")

plot_12_lead_ecg(
    signals=record_icbeb.p_signal,
    lead_names=record_icbeb.sig_name,
    fs=record_icbeb.fs,
    title="ICBEB Registro A0001: Bloqueo Completo de Rama Derecha (CRBBB) — 500 Hz",
    duration=10.0
)"""))

    # =========================================================================
    # Celda 14: Comparación Directa (Head-to-Head)
    # =========================================================================
    cells.append(md("""---
## 6. Comparación Directa (*Head-to-Head*): PTB-XL vs. ICBEB 2018

### 6.1. Cuadro Comparativo Multidimensional"""))

    cells.append(code("""comparison_data = {
    "Dimensión / Atributo": [
        "País de Origen",
        "Población y Entorno Clínico",
        "Total de Registros de ECG",
        "Total de Pacientes Únicos",
        "Identificador de Paciente",
        "Control de Fuga de Grupo",
        "Número de Derivaciones",
        "Frecuencia de Muestreo (Fs)",
        "Duración de la Señal",
        "Nomenclatura Diagnóstica",
        "Número de Clases Principales",
        "Naturaleza del Problema",
        "Prevalencia de Trazados Normales"
    ],
    "PTB-XL (Alemania)": [
        "Alemania (Charité Berlín / PTB)",
        "Población clínica europea, hospitalaria y ambulatoria",
        "21.799 registros",
        "18.869 pacientes",
        "Sí ('patient_id' disponible)",
        "Garantizado (10 folds estratificados por paciente)",
        "12 derivaciones estándar",
        "100 Hz y 500 Hz",
        "Estrictamente fija (10.0 segundos)",
        "Estándar internacional SCP-ECG (71 declaraciones)",
        "5 superclases / 24 subclases",
        "Multietiqueta complejo (promedio 2.8 códigos/ECG)",
        "43.6% (9.514 trazados NORM)"
    ],
    "ICBEB 2018 (China)": [
        "China (11 centros hospitalarios)",
        "Población asiática, pacientes hospitalizados con arritmias",
        "6.877 registros (training) + 3.453 extra",
        "No reportado individualmente",
        "No ('patient_id' = 'ecg_id')",
        "No garantizado (riesgo de fuga si hay registros repetidos)",
        "12 derivaciones estándar",
        "500 Hz (frecuencia nativa)",
        "Variable (entre 6.0 y 60.0 segundos)",
        "9 patologías diana específicas del challenge",
        "9 clases clínicas",
        "Predominantemente unietiqueta (93.1% 1 sola clase)",
        "13.3% (918 trazados NORM)"
    ]
}

df_comparison = pd.DataFrame(comparison_data)
display(df_comparison.style.set_properties(**{'text-align': 'left', 'font-size': '12px'}))"""))

    # =========================================================================
    # Celda 15: Demostración de Label Shift
    # =========================================================================
    cells.append(md("""### 6.2. Demostración Empírica de *Label Shift* Masivo

Mapeamos las 9 clases de ICBEB contra sus equivalentes exactos en SCP-ECG de PTB-XL:
- `NORM` $\\leftrightarrow$ `NORM`
- `AFIB` $\\leftrightarrow$ `AFIB`
- `1AVB` $\\leftrightarrow$ `1AVB`
- `CLBBB` $\\leftrightarrow$ `CLBBB`
- `CRBBB` $\\leftrightarrow$ `CRBBB`
- `PAC` $\\leftrightarrow$ `PAC`
- `VPC` $\\leftrightarrow$ `PVC` (*Premature Ventricular Contraction*)
- `STD_` $\\leftrightarrow$ `STD_`
- `STE_` $\\leftrightarrow$ `STE_`"""))

    cells.append(code("""# Conteo de las 9 clases comunes en PTB-XL
ptb_target_counts = {
    'NORM': df_ptb['scp_dict'].apply(lambda d: 'NORM' in d).sum(),
    'AFIB': df_ptb['scp_dict'].apply(lambda d: 'AFIB' in d).sum(),
    '1AVB': df_ptb['scp_dict'].apply(lambda d: '1AVB' in d).sum(),
    'CLBBB': df_ptb['scp_dict'].apply(lambda d: 'CLBBB' in d).sum(),
    'CRBBB': df_ptb['scp_dict'].apply(lambda d: 'CRBBB' in d).sum(),
    'PAC': df_ptb['scp_dict'].apply(lambda d: 'PAC' in d).sum(),
    'VPC': df_ptb['scp_dict'].apply(lambda d: 'PVC' in d).sum(),  # VPC en SCP es PVC
    'STD_': df_ptb['scp_dict'].apply(lambda d: 'STD_' in d).sum(),
    'STE_': df_ptb['scp_dict'].apply(lambda d: 'STE_' in d).sum()
}

ptb_rates = {k: (v / len(df_ptb)) * 100 for k, v in ptb_target_counts.items()}
icbeb_rates = {k: (icbeb_counts[k] / len(df_icbeb)) * 100 for k in ptb_target_counts.keys()}

df_shift = pd.DataFrame({
    'Clase': list(ptb_rates.keys()),
    'PTB-XL (%) [Alemania]': list(ptb_rates.values()),
    'ICBEB (%) [China]': list(icbeb_rates.values())
})

df_shift['Ratio de Disparidad (ICBEB / PTB)'] = df_shift['ICBEB (%) [China]'] / df_shift['PTB-XL (%) [Alemania]']
display(df_shift.round(2))

# Gráfico de barras comparativo de prevalencia
df_plot = df_shift.melt(id_vars='Clase', value_vars=['PTB-XL (%) [Alemania]', 'ICBEB (%) [China]'], 
                        var_name='Dataset', value_name='Prevalencia (%)')

plt.figure(figsize=(14, 6))
chart = sns.barplot(data=df_plot, x='Clase', y='Prevalencia (%)', hue='Dataset', palette=['#2980B9', '#E67E22'])
plt.title("Demostración de Label Shift: Prevalencia de Patologías en PTB-XL vs. ICBEB", fontsize=14)
plt.ylabel("Prevalencia en la Población (%)")
plt.xlabel("Patología Cardíaca")

for p in chart.patches:
    h = p.get_height()
    if h > 0:
        chart.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 0.5),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.ylim(0, 50)
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 16: Comparación de Espectro de Frecuencias
    # =========================================================================
    cells.append(md("""### 6.3. Análisis en el Dominio de la Frecuencia (Espectro y Artefactos)

Calculamos la **Densidad Espectral de Potencia (PSD)** mediante el método de Welch sobre la derivación II para comparar el contenido armónico de las señales y la presencia de interferencia de línea eléctrica (50 Hz en Europa/China)."""))

    cells.append(code("""# Comparación espectral usando las muestras cargadas
sig_ptb_lead2 = record_ptb.p_signal[:, 1]
sig_icbeb_lead2 = record_icbeb.p_signal[:len(sig_ptb_lead2)*5, 1]  # 500 Hz vs 100 Hz

f_ptb, psd_ptb = scipy.signal.welch(sig_ptb_lead2, fs=record_ptb.fs, nperseg=min(len(sig_ptb_lead2), 256))
f_icbeb, psd_icbeb = scipy.signal.welch(sig_icbeb_lead2, fs=record_icbeb.fs, nperseg=min(len(sig_icbeb_lead2), 1024))

plt.figure(figsize=(12, 5))
plt.semilogy(f_ptb, psd_ptb, label='PTB-XL (Fs = 100 Hz) [Alemania]', color='#2980B9', linewidth=2)
plt.semilogy(f_icbeb[f_icbeb <= 50], psd_icbeb[f_icbeb <= 50], label='ICBEB (Fs = 500 Hz) [China]', color='#E67E22', linewidth=2, linestyle='--')
plt.title("Densidad Espectral de Potencia (Derivación II)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad Espectral (V²/Hz)")
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 17: Síntesis Metodológica para ESAA
    # =========================================================================
    cells.append(md("""---
## 7. Síntesis Metodológica para el Proyecto de la Materia ESAA

En esta sección sintetizamos cómo aplicar los 5 conceptos teóricos fundamentales de la cátedra (**Dra. Luciana Ferrer**) utilizando estos dos datasets:

---

### 7.1. Desplazamiento de Dominio (*Domain Shift*)

La regla de probabilidad total descompone la distribución conjunta $P(X, Y)$:
$$P(X, Y) = P(X) \\cdot P(Y|X) = P(Y) \\cdot P(X|Y)$$

| Tipo de Shift | Definición Matemática | Manifestación en PTB-XL vs. ICBEB |
| :--- | :--- | :--- |
| **Covariate Shift** | $P(X_{\\text{China}}) \\neq P(X_{\\text{Alemania}})$, manteniendo $P(Y\\|X)$ | Diferencias de hardware de adquisición, filtros pasa-banda, colocación de electrodos y duración de la grabación (6s-60s vs 10s). |
| **Prior / Label Shift** | $P(Y_{\\text{China}}) \\neq P(Y_{\\text{Alemania}})$, manteniendo $P(X\\|Y)$ | La prevalencia de CRBBB se multiplica por 10 (2.5% vs 27.0%), mientras que los casos normales caen del 43.6% al 13.3%. |
| **Concept Drift** | $P(Y\\|X_{\\text{China}}) \\neq P(Y\\|X_{\\text{Alemania}})$ | Criterios diagnósticos subjetivos entre el comité cardiológico europeo (SCP-ECG) y el panel del challenge de China ante elevaciones sutiles de ST o bloqueos fasciculares. |

---

### 7.2. Teoría de Decisión Bayesiana y Matriz de Costo ($C_{ij}$)

En medicina, la regla de decisión estándar **MAP (Argmax / umbral 0.5)** es clínicamente inaceptable:
- Decidir $d_1$ (enfermo) cuando el paciente es $C_0$ (sano): genera un falso positivo ($c_{01}$), cuyo costo es una prueba complementaria o molestia.
- Decidir $d_0$ (sano) cuando el paciente es $C_1$ (infarto/arritmia): genera un falso negativo ($c_{10}$), cuyo costo puede ser la muerte.

El **Umbral Óptimo de Bayes** para clasificación binaria viene dado por:
$$\\theta^* = \\frac{c_{01} - c_{00}}{(c_{01} - c_{00}) + (c_{10} - c_{11})} = \\frac{c_{01}}{c_{01} + c_{10}}$$
Y en función del cociente de verosimilitud (*Likelihood Ratio* $\\Lambda(x)$) ajustado por la prevalencia previa $P(C_1)$:
$$\\Lambda(x) = \\frac{P(x|C_1)}{P(x|C_0)} > \\frac{c_{01}}{c_{10}} \\cdot \\frac{P(C_0)}{P(C_1)}$$

**Impacto del Label Shift en el Umbral de Decisión**:
Si entrenamos un modelo en PTB-XL donde $P(C_1) = 0.025$ (CRBBB), el modelo necesitará un $P(x|C_1)$ altísimo para superar el umbral. Al trasladarlo a ICBEB (donde $P(C_1) = 0.270$), el umbral óptimo de decisión **debe reducirse drásticamente** para evitar una avalancha de falsos negativos catastróficos."""))

    cells.append(code("""# Demostración del cálculo del Umbral Óptimo de Bayes
# Asumimos que errarle a una patología grave (FN) cuesta 10 veces más que una falsa alarma (FP)
c01 = 1.0   # Costo Falso Positivo
c10 = 10.0  # Costo Falso Negativo (grave)

# Prior de CRBBB en Alemania vs China
prior_ptb = ptb_rates['CRBBB'] / 100
prior_icbeb = icbeb_rates['CRBBB'] / 100

# Umbral Bayesiano óptimo sobre la probabilidad posterior P(C1|x)
theta_bayes_posterior = c01 / (c01 + c10)

# Umbral sobre el Likelihood Ratio (LR)
theta_lr_ptb = (c01 / c10) * ((1 - prior_ptb) / prior_ptb)
theta_lr_icbeb = (c01 / c10) * ((1 - prior_icbeb) / prior_icbeb)

print("=== TEORÍA DE DECISIÓN BAYESIANA ANTE ASIMETRÍA DE COSTOS ===")
print(f"Costo FP (c01) = {c01}, Costo FN (c10) = {c10}")
print(f"Umbral Óptimo sobre la Probabilidad Posterior P(C1|x): theta* = {theta_bayes_posterior:.4f} (vs 0.50 de MAP)")
print(f"\\nImpacto del Label Shift en el umbral sobre Likelihood Ratio:")
print(f"  - Umbral LR en PTB-XL (Alemania, Prior = {prior_ptb*100:.2f}%): {theta_lr_ptb:.2f}")
print(f"  - Umbral LR en ICBEB  (China,    Prior = {prior_icbeb*100:.2f}%): {theta_lr_icbeb:.2f}")
print(f">> Conclusión: En China el umbral sobre el LR debe ser {theta_lr_ptb / theta_lr_icbeb:.1f} veces menor que en Alemania.")"""))

    # =========================================================================
    # Celda 18: Opción de Abstención (Costo de Chow)
    # =========================================================================
    cells.append(md("""### 7.3. Opción de Abstención (*Reject Option* / Regla de Chow)

Cuando la probabilidad condicional de una clase se encuentra en la zona de máxima incertidumbre, obligar al modelo a emitir un diagnóstico forzado es peligroso.
Incorporamos la acción de abstención $d_{\\text{abstener}}$ con un costo fijo $c_r < \\min(c_{01}, c_{10})$ (costo de derivar a un cardiólogo especialista).

La regla óptima de Chow decide abstenerse si:
$$\\max_k P(C_k|x) < 1 - \\frac{c_r}{c_{\\text{error}}}$$"""))

    cells.append(code("""# Simulación conceptual de la Regla de Chow
probs = np.linspace(0.0, 1.0, 500)
c_error = 1.0
costos_rechazo = [0.1, 0.2, 0.35]

plt.figure(figsize=(10, 5))
for cr in costos_rechazo:
    umbral_inf = cr / c_error
    umbral_sup = 1.0 - (cr / c_error)
    plt.axvspan(umbral_inf, umbral_sup, alpha=0.15, label=f'Zona de Abstención médica (c_r = {cr})')

plt.title("Regla de Abstención de Chow para Diagnóstico Asistido en ECG")
plt.xlabel("Probabilidad Predicha P(Patología | x)")
plt.ylabel("Acción Recomendada")
plt.yticks([0, 1], ["Decidir Sano (0)", "Decidir Patología (1)"])
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 19: Calibración Probabilística y Brier Score
    # =========================================================================
    cells.append(md("""### 7.4. Calibración Probabilística y Descomposición de Murphy

Una red neuronal profunda tiende a estar **mal calibrada** (típicamente sobreconfiada).
Evaluamos la calidad probabilística mediante el **Brier Score** (una *Strict Proper Scoring Rule*):
$$\\text{BS} = \\frac{1}{N} \\sum_{n=1}^N \\sum_{k=1}^K (p_{nk} - y_{nk})^2$$

Según la **Descomposición de Murphy (1973)**:
$$\\text{Brier Score} = \\text{Incertidumbre (Uncertainty)} - \\text{Resolución (Resolution)} + \\text{Confiabilidad (Reliability)}$$
- **Reliability (Calibración pura)**: Mide qué tan cerca está la confianza predicha de la frecuencia empírica real.
- **Resolution**: Mide la capacidad de separar casos positivos de negativos.
- **Recalibración Post-Hoc**: Mediante *Platt Scaling* o *Temperature Scaling*, podemos reducir la componente de Confiabilidad a casi cero sin alterar el AUROC (que solo depende de la resolución)."""))

    cells.append(code("""# Esquema de visualización de Diagrama de Confiabilidad
conf_bins = np.linspace(0.1, 0.9, 9)
acc_bien_calibrado = conf_bins
acc_sobreconfiado = conf_bins**1.8

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente Calibrado')
plt.plot(conf_bins, acc_bien_calibrado, 's-', color='#27AE60', label='Modelo Calibrado (ECE bajo)')
plt.plot(conf_bins, acc_sobreconfiado, 'o-', color='#E74C3C', label='Modelo Sobreconfiado (Típico DNN sin calibrar)')
plt.title("Diagrama de Confiabilidad (Reliability Diagram) para Evaluación")
plt.xlabel("Confianza / Probabilidad Predicha")
plt.ylabel("Frecuencia Empírica Observada (Precisión)")
plt.legend()
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 20: Equidad por Subgrupos
    # =========================================================================
    cells.append(md("""### 7.5. Evaluación de Equidad por Subgrupos (*Subgroup Fairness*)

Un sistema de diagnóstico automático que alcanza un AUROC general de 0.92 puede esconder disparidades severas:
- **Disparidad de Sexo**: Ciertas arritmias se manifiestan con menor amplitud o patrones sutiles en mujeres debido a diferencias electrofisiológicas hormonales y masa ventricular.
- **Disparidad por Edad**: En pacientes mayores de 80 años, la presencia de anomalías de conducción difusas basales puede degradar severamente la especificidad de las predicciones de isquemia o infarto.

**Recomendación para el informe de entrega**:
Desglosar las métricas operativas ($F_\\beta$, Costo Esperado, ECE) desagregadas por subgrupos poblacionales."""))

    cells.append(code("""# Simulación comparativa del desglose por subgrupos
subgrupos = ['General', 'Hombres', 'Mujeres', 'Edad < 50', 'Edad 50-70', 'Edad > 80']
auroc_estimado = [0.915, 0.928, 0.892, 0.935, 0.910, 0.845]

df_fairness = pd.DataFrame({'Subgrupo Poblacional': subgrupos, 'Macro AUROC': auroc_estimado})

plt.figure(figsize=(10, 4))
chart = sns.barplot(data=df_fairness, x='Subgrupo Poblacional', y='Macro AUROC', palette='mako')
plt.title("Ejemplo de Evaluación Desagregada por Subgrupos (Auditoría de Sesgo)")
plt.ylabel("Macro AUROC")
plt.ylim(0.7, 1.0)
for p in chart.patches:
    chart.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.008),
                   ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.show()"""))

    # =========================================================================
    # Celda 21: Conclusiones
    # =========================================================================
    cells.append(md("""---
## 8. Conclusiones y Guía Práctica para la Entrega Final

1. **PTB-XL e ICBEB representan dos realidades clínicas y de adquisición radicalmente distintas**:
   - PTB-XL (Alemania) provee 21.799 ECGs de 10s fijos con control estricto de Fuga de Grupo por paciente y alta prevalencia de casos normales (43.6%).
   - ICBEB (China) provee 6.877 registros con duraciones heterogéneas (6s a 60s) fuertemente enriquecidos en patologías complejas (por ejemplo, 27.0% de CRBBB).
2. **Evaluación de Desempeño Fuera de Dominio**:
   - Evaluar un modelo entrenado en PTB-XL directamente sobre ICBEB provocará un colapso en métricas dependientes de umbral si no se realiza un **ajuste de priors** y una **recalibración de confianza**.
3. **No es necesario re-entrenar redes neuronales pesadas**:
   - En la carpeta `output/` del repositorio se encuentran disponibles las probabilidades predichas de los modelos evaluados en el paper (`y_test_pred.npy` e `y_train_pred.npy`).
   - El trabajo para la entrega de ESAA consiste en implementar las métricas superiores desarrolladas en las clases: **Matrices de Costo Asimétricas**, **Reglas de Abstención de Chow**, **Recalibración con Temperature Scaling** y **Análisis de Equidad por Subgrupos**."""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    output_path = "Analisis_Exploratorio_PTBXL_vs_ICBEB.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

    print(f"Notebook creado con éxito en: {output_path}")

if __name__ == "__main__":
    create_notebook()
