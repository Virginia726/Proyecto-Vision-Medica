# Proyecto-Vision-Medica
Desarrollo de una aplicación para la detección de anomalías médicas usando visión artificial 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Ecosistema-EE4C2C.svg)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Torchvision-SwinT%20%7C%20DeepLabV3%2B%20%7C%20FasterRCNN-green.svg)](https://pytorch.org/vision/stable/index.html)
[![Verano Científico](https://img.shields.io/badge/Proyecto-Verano%20Cient%C3%ADfico%202026-brightgreen.svg)]()

Este repositorio contiene la arquitectura backend y la suite de algoritmos para el análisis automatizado de radiografías de tórax. El proyecto integra un enfoque modular que resuelve tres tareas fundamentales en la visión artificial médica: **Clasificación Multietiqueta**, **Segmentación Anatómica** y **Localización Geométrica de Lesiones**, todo optimizado mediante **Carga Perezosa (*Lazy Loading*)** para ejecutar pipelines estables en entornos locales con restricciones de memoria y hardware.

---

## Tabla de Contenidos

* [Descripción del Proyecto](#-descripción-del-proyecto)
* [Arquitectura del Repositorio](#-arquitectura-del-repositorio)
* [Módulos del Sistema](#-módulos-del-sistema)
  * [1. Clasificación Multietiqueta (Swin Transformer)](#1-clasificación-multietiqueta-swin-transformer)
  * [2. Segmentación Semántica (DeepLabV3+)](#2-segmentación-semántica-deeplabv3)
  * [3. Detección y Localización (Faster R-CNN)](#3-detección-y-localización-faster-r-cnn)
* [Bitácora de Control de Fallas y Optimización de Hardware](#-bitácora-de-control-de-fallas-y-optimización-de-hardware)
* [Resultados y Métricas de Control](#-resultados-y-métricas-de-control)
* [Instalación y Uso](#-instalación-y-uso)
* [Trabajo Futuro](#-trabajo-futuro)
* [Referencias y Licencia](#-referencias-y-licencia)

---

##  Descripción del Proyecto

El diagnóstico oportuno en radiología torácica enfrenta una alta carga de trabajo clínico. Este sistema propone una solución asistida por visión artificial estructurada en 3 niveles de análisis:
1. **Clasificación:** ¿Qué patologías o condiciones clínicas están presentes simultáneamente?
2. **Segmentación:** ¿Dónde están delimitados exactamente los campos pulmonares?
3. **Detección:** ¿Cuáles son las coordenadas específicas (cajas delimitadoras) de las anomalías u opacidades?

---

##  Arquitectura del Repositorio

El proyecto mantiene un desacoplamiento estricto entre la capa de ingestión diferida de datos, el código fuente (`src/`), los módulos de entrenamiento y la persistencia de checkpoints (`.pth`):

```text
Proyecto-Vision-Medica/
├── data/
│   ├── clasificacion/
│   │   ├── images/                           # Radiografías originales para clasificación
│   │   ├── masks/                            # Máscaras asociadas
│   │   ├── crear_csv.py                      # Script de generación del maestro de etiquetas
│   │   └── labels.csv                        # Archivo CSV de etiquetas clínicas
│   ├── deteccion\train/
│   │   └── dataset.py                        # Custom Dataset para el pipeline de detección
│   └── segmentacion/                         # Datos e imágenes de segmentación anatómica
├── models/
│   └── modelo_clasificacion.pth             # Checkpoint entrenado del clasificador
├── src/
│   ├── deteccion/                            # Módulo fuente de detección
│   └── segmentacion/                         # Módulo fuente de segmentación
├── deeplabv3_pulmon_epoch_1.pth              # Checkpoint de segmentación (Época 1)
├── figura_radiografia_caja_delimitadora.png  # Gráfica de salida con bounding boxes predichas
├── modelo_deteccion_simulador.pth            # Pesos del simulador/emulador de detección
├── swin_t_ligero.pth                         # Checkpoint ligero de Swin Transformer
├── swin_transformer_checkpoint_epoch_1.pth   # Checkpoints intermedios por época
├── swin_transformer_checkpoint_epoch_2.pth
├── swin_transformer_checkpoint_epoch_3.pth
├── train_deteccion.py                        # Pipeline principal de Detección
├── train_segmentacion.py                     # Pipeline principal de Segmentación
├── train.py                                  # Pipeline principal de Clasificación
└── README.md                                 # Documentación principal del sistema
