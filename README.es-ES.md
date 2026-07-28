# Traducción Automática Neuronal Seq2Seq desde Cero
Implementación desde cero de Seq2Seq + Atención de Bahdanau para traducción automática neuronal (Alemán → Inglés, conjunto de datos Multi30k)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![BLEU](https://img.shields.io/badge/BLEU-56.3-brightgreen)](https://huggingface.co/spaces/xu2409324124/lstm-translator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="model_architecture_bahdanau_lstm.png" alt="Bahdanau Attention + LSTM Seq2Seq Architecture" width="800"/>
  <br>
  <em>Diagrama de arquitectura del modelo Bahdanau (Aditiva) Attention + LSTM Seq2Seq (incluye inversión de la frase origen + mecanismo de input feeding)</em>
</p>

## Aspectos Destacados del Proyecto

- Reproducción fiel de las técnicas centrales de Sutskever et al. (2014): **Inversión de la frase origen** + Atención Aditiva (Bahdanau Attention).
- Codificador/Decodificador LSTM (hidden=256~512 ajustable, dropout=0.4, label smoothing=0.1).
- Optimización del entrenamiento: Decaimiento dinámico de teacher forcing, recorte de gradientes (gradient clipping), AdamW, programación de tasa de aprendizaje, early stopping + monitoreo de validación.
- **Visualización de mapas de calor de atención** (Matplotlib/Seaborn), permitiendo análisis interpretable y diagnóstico de errores.
- Decodificación **Beam Search** (size=3~5, con soporte para penalización de longitud).
- Evaluación estandarizada con sacreBLEU.
- **Interfaz de traducción en tiempo real con Gradio** (desplegada en Hugging Face Spaces).

## Rendimiento Final (Multi30k test set, sacreBLEU)

| Configuración                          | Puntuación BLEU | Notas                                     |
|---------------------------------------|-----------------|-------------------------------------------|
| Greedy decoding (best epoch)          | **56.3**       | Epoch 35, mejor checkpoint                |
| Beam search (size=3~5)               | ~54–57         | Fluctuaciones ligeras, optimizable         |
| Baseline sin inversión de frase origen | ~30–35         | Valida que el "reverse trick" mejora significativamente |

> Alcanzar un BLEU de 56+ en Multi30k con solo ~29k pares de entrenamiento supera notablemente la mayoría de los tutoriales de seq2seq de código abierto y las implementaciones base de 2014–2017.

## Curva de Loss (Entrenamiento 30+ epochs)

<p align="center">
  <img src="loss_curve_lstm.png" alt="Training & Validation Loss Curve" width="700"/>
  <br>
  <em>Azul: Loss de entrenamiento　Rojo: Loss de validación　Mejor Loss de validación: 4.3435 (disparó early stopping)</em>
</p>

## Ejemplo de Mapa de Calor de Atención (Greedy + Source Reversed)

<p align="center">
  <img src="attention_heatmap_example.jpeg" alt="Attention Heatmap Example" width="700"/>
  <br>
  <em>Eje X: Frase origen en alemán (invertida)　Eje Y: Frase generada en inglés　La intensidad del color indica el peso de atención</em>
</p>

## Demostración de Traducción en Tiempo Real

Pruébalo directamente en el navegador (soporta cualquier frase en alemán):

👉 **[LSTM Translator on Hugging Face Spaces](https://huggingface.co/spaces/xu2409324124/lstm-translator)**

## Cómo Ejecutar

### Requisitos
- Python 3.8+
- PyTorch 2.0+ (se recomienda CUDA)
- GPU: RTX 4060 o superior (Soporte de 8GB+ VRAM para batch=64~128)

### Pasos
1. Clonar el repositorio
   ```bash
   git clone https://github.com/2409324124/seq2seq-nmt-from-scratch.git
   cd seq2seq-nmt-from-scratch
   ```

2. Instalar dependencias
   ```bash
   pip install -r requirements.txt
   ```

3. Entrenar
   ```bash
   python train.py
   ```

4. Probar y calcular BLEU
   ```bash
   python translate.py --mode test --beam 5
   ```

5. Iniciar interfaz de Gradio
   ```bash
   python translate_gradio.py
   ```

El conjunto de datos Multi30k (en-de) se descarga automáticamente desde Hugging Face.

## Historia de Autoaprendizaje y Agradecimientos

Este proyecto fue realizado por un **estudiante autodidacta desde cero y no especializado en Ciencias de la Computación (trasfondo en Sociología)**, mediante guías conversacionales con modelos de lenguaje extenso, recorriendo todo el camino: configuración del entorno → Introducción a PyTorch → MNIST CNN → GRU Seq2Seq → LSTM + Attention.

Agradecimientos:
- Documentación y tutoriales oficiales de PyTorch.
- bentrevett/pytorch-seq2seq (referencia clásica).
- Hugging Face Datasets & Spaces.
- Librería sacreBLEU.

¡Bienvenido a hacer fork, star o abrir issues! También se agradecen discusiones sobre optimizaciones (como bidirectional encoder, multi-head attention, pretrained embeddings).

Happy translating! 🚀

---

### Descripción Detallada del Módulo de Atención Aditiva

El módulo de atención aditiva (Additive Attention, también conocido como **Bahdanau Attention**) es uno de los componentes centrales de toda la arquitectura Seq2Seq, logrando la alineación dinámica de información entre el codificador (Encoder) y el decodificador (Decoder). Se basa en el artículo clásico de 2015 de Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", un mecanismo emblemático en las primeras etapas de la traducción automática neuronal (NMT).

#### 1. Flujo General

- **Entradas**:
  - **Query**: Estado oculto previo del decodificador, forma: `(batch_size, hidden_size)`.
  - **Keys / Values**: Todas las secuencias de salida del codificador (`encoder_outputs`), forma: `(batch_size, src_len, hidden_size * 2)` (LSTM bidireccional).

- **Pasos de Cálculo**:
  1. Cálculo de la puntuación de energía: Proyección lineal de query y keys, fusión aditiva + activación tanh.
  2. Obtención de la puntuación original mediante una capa lineal.
  3. Normalización mediante softmax para obtener los pesos de atención.
  4. Suma ponderada para obtener el vector de contexto.
  5. Concatenación del vector de contexto con el embedding de entrada actual, introduciéndolo al LSTM del decoder.

- **Salidas**:
  - **Context vector**: forma `(batch_size, hidden_size)`.
  - **Attention weights**: forma `(batch_size, src_len)` (utilizado para la visualización del mapa de calor).

Este proceso se ejecuta en cada paso de tiempo (time step) de la decodificación, permitiendo que el modelo se enfoque dinámicamente en la información más relevante de la frase origen.

El diagrama de arquitectura (`model_architecture_bahdanau_lstm.png`) muestra claramente este proceso:
- Izquierda: Encoder outputs → Linear (Ua) → Add (con Linear wa proveniente del previous decoder hidden) → σ (softmax) → weighted sum → context.
- Derecha: Context + word embedding → AttentionConcat → LSTM Decoder → Linear → Output Logits.

#### 2. Fórmulas de Cálculo (Detalles Matemáticos Core)

Bahdanau Attention utiliza la fusión aditiva de las proyecciones de query y keys, siguiendo estrictamente el artículo original (Bahdanau et al., 2015).

- **Cálculo de Energía (Alignment/Energy score)**:

$$
e_{ij} = v_a^\top \tanh \left( W_a s_{i-1} + U_a h_j \right)
$$

  - $s_{i-1}$: Estado oculto del decoder en el instante anterior (query).
  - $h_j$: J-ésimo estado oculto del encoder (annotation/key).
  - $W_a, U_a$: Matrices de proyección aprendibles.
  - $v_a$: Vector de pesos aprendible (vector columna, producto punto con el resultado de tanh tras la transposición).

- **Pesos de Atención (Alignment weights)**:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

Se realiza una normalización softmax sobre las puntuaciones de energía de todas las posiciones de origen para asegurar que la suma de los pesos sea 1.

- **Vector de Contexto (Context vector)**:

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

El vector de contexto es la suma ponderada de los estados ocultos de origen, utilizado directamente en el siguiente cálculo del decoder.

- **Fusión en el Decoder** (método de input feeding, adoptado en mi implementación):

```python
lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
```

El módulo de atención reside como una clase independiente `BahdanauAttention`, integrada en el forward de `AttnDecoderLSTM`:

- **Inicialización**:
  ```python
  self.attention = BahdanauAttention(hidden_size)  # Clase de atención personalizada
  self.lstm = nn.LSTM(hidden_size * 3, hidden_size, ...)  # La dimensión de entrada implica la concatenación de context + embedding
  ```

- **Cálculo Forward**:
  ```python
  context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # query = hidden[0]
  lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # input feeding
  output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
  ```

- **Núcleo de la clase BahdanauAttention**:
  - Tres capas lineales: `Wa` (proyección query), `Ua` (proyección keys, soporta hidden*2 bidireccional), `Va` (scores).
  - Energy: `tanh(Wa(query) + Ua(keys))`.
  - Scores: `Va(energy)`.
  - Weights: `softmax(scores)`.
  - Context: `bmm(weights, encoder_outputs)`.

Soporta codificadores bidireccionales para mejorar el rendimiento y utiliza el mecanismo de input feeding para fortalecer la capacidad de percepción de contexto del decoder.

#### 4. Ventajas y Funciones

- **¿Por qué elegir atención aditiva?**  
  El Seq2Seq tradicional depende solo del estado final del codificador, lo que suele provocar la pérdida de información en secuencias largas (cuello de botella de información). La atención aditiva permite que el decoder consulte dinámicamente la frase origen, mejorando significativamente las dependencias a larga distancia (como referencias pronominales y alineación gramatical). En el conjunto de datos Multi30k, ayudó a elevar el modelo de un baseline de ~30 BLEU a **56.3**.

- **Aditiva vs. Otras Atenciones**:
  - A diferencia de la atención de producto punto (dot-product) de Luong, la aditiva utiliza tanh + fusión lineal, lo que la hace más flexible y con más parámetros, siendo especialmente adecuada para conjuntos de datos pequeños como Multi30k.
  - Complejidad computacional: O(src_len × hidden²), lo cual es totalmente aceptable para hidden=256 y una RTX 4060.

- **Función específica en este modelo**:
  - Mejora la calidad de traducción: los pesos de atención capturan la alineación a nivel de palabra (ej. sustantivo alemán → palabra correspondiente en inglés).
  - Soporta interpretabilidad: la generación de mapas de calor mediante `attn_weights` facilita el debugging y el análisis (como verificar si la atención en frases largas es uniforme).
  - Combinación con inversión de frase origen: la inversión fortalece las dependencias de corto alcance, y la atención optimiza aún más el rendimiento en frases largas.

- **Limitaciones Potenciales**:
  - Atención Global: En cada paso se atiende a toda la frase origen; la carga computacional crece linealmente si `src_len` es muy largo (sin problema en Multi30k, pero requeriría optimización para WMT).
  - Atención de Cabeza Única: Actualmente es de una sola cabeza; en el futuro podría extenderse a multi-head attention para capturar diferentes patrones (gramaticales vs. semánticos).
