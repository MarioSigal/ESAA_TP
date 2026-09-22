# Resumen: Benchmark de ECG con PTB-XL
**Proyecto:** [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking)  
**Materia:** Evaluación de Sistemas de Aprendizaje Automático (ESAA)

---

### ¿De qué se trata el proyecto?

Básicamente, este repositorio arma una **competencia de modelos de Deep Learning** para ver cuál es el mejor diagnosticando problemas cardíacos en electrocardiogramas (ECG) de 12 derivaciones.

* **Los datos:** Usa **PTB-XL**, una base de datos pública con casi 22.000 registros de ECG de unos 19.000 pacientes.
* **El problema:** Es multietiqueta (un paciente puede tener varias cosas a la vez: un infarto, una arritmia, etc.).
* **Los modelos:** Compara varias redes neuronales modernas (ResNet 1D, Inception 1D, LSTMs) contra modelos más básicos.

---

### ¿Cómo evalúan los modelos hoy?

1. **Partición de datos:** Bien hecha. Dividieron los datos cuidando que todos los ECGs de un mismo paciente queden en el mismo grupo. Así el modelo no "hace trampa" reconociendo al paciente en vez de a la enfermedad.
2. **La métrica que usan:** Usan casi exclusivamente **AUROC**. Es un número general que mide qué tan bien separa los casos positivos de los negativos.
3. **Incertidumbre:** Hacen 100 remuestreos (*Bootstrap*) en el conjunto de test para dar un margen de error (los percentiles 5% y 95%).

---

### ¿Qué cosas podemos hacer nosotros para mejorar la evaluación?

Lo que hace el repo hoy es muy básico: solo mira un número general (el AUROC) y no evalúa cómo funcionaría el modelo en un hospital de verdad. 

Acá van **5 cosas muy concretas y fáciles de entender que podemos hacer** (basadas en lo que vimos en las clases):

1. **Ponerle costo a los errores (no todos los errores son iguales):**
   * *La idea:* Hoy el modelo trata igual a cualquier error. Pero en la vida real, que el modelo mande a hacer un estudio de más a alguien sano es una molestia; pero que mande a su casa a alguien que se está infartando **es gravísimo**.
   * *Qué hacemos:* Armamos una matriz de costos donde errarle a un infarto cueste mucho más caro, y medimos qué modelo genera menor costo total (*Costo Esperado* de la Clase 3).

2. **Dejar que el modelo diga "No sé, que lo revise un médico" (Abstención):**
   * *La idea:* En medicina, si el algoritmo no está seguro, obligarlo a tirar una moneda al aire es peligroso.
   * *Qué hacemos:* Le permitimos al modelo abstenerse cuando tenga dudas (el *Costo de Chow* de la Clase 3). Evaluamos qué tan bueno es cuando solo decide sobre los casos donde está seguro.

3. **Ver si el modelo "delira confianza" y corregirlo (Calibración):**
   * *La idea:* Las redes neuronales suelen ser muy agrandadas: capaz te dicen *"estoy 99% seguro de que hay infarto"* cuando en realidad la probabilidad real es del 70%.
   * *Qué hacemos:* Medimos si las probabilidades son honestas (usando el *Brier Score* de la Clase 5) y le aplicamos un ajuste simple (*Platt o Temperature Scaling* de la Clase 6) para calibrar su confianza.

4. **Ver si anda peor en mujeres o ancianos (Análisis por subgrupos):**
   * *La idea:* Como vimos en la Clase 1, un sistema puede dar un promedio excelente a nivel general, pero fallar estrepitosamente en una minoría.
   * *Qué hacemos:* Como el dataset tiene edad y sexo, separamos las predicciones y nos fijamos: *¿el modelo rinde igual de bien en hombres que en mujeres? ¿falla más en personas mayores de 80 años?*

5. **Probar si funciona en otro hospital (Dataset Shift):**
   * *La idea:* El modelo fue entrenado con datos de un centro de salud de Alemania. Como vimos en la Clase 2, los modelos suelen caerse cuando cambian de entorno.
   * *Qué hacemos:* Agarras el modelo ya entrenado y lo testeas directo sobre el dataset de China (ICBEB) para ver si de verdad aprendió cardiología o si solo memorizó las máquinas del hospital alemán.

---

### Conclusión práctica para la entrega

Lo mejor de todo esto es que **no hace falta entrenar ningún modelo ni gastar horas de GPU**: en la carpeta `output/` del repo ya están guardadas las respuestas que dio cada modelo (`y_test_pred.npy`). 

Nuestro trabajo para la materia consiste simplemente en agarrar esos archivos con un script de Python y calcularles estas métricas más inteligentes.
