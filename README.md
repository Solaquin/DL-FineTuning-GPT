# DL Finetuning GPT

Un proyecto de fine-tuning del modelo TinyLlama especializado en contenido gaming utilizando LoRA (Low-Rank Adaptation).

## 📋 Descripción

Este repositorio contiene la implementación de un modelo de lenguaje optimizado para el dominio gaming, basado en TinyLlama-1.1B y entrenado con técnicas de LoRA para eficiencia computacional.

## 🎮 Presentación del Proyecto

Puedes ver la presentación completa del proyecto aquí:
[Ver Presentación en Canva](https://www.canva.com/design/DAG5qjyqC4I/QeI0sOQymluWi2aIfQEB_g/view?utm_content=DAG5qjyqC4I&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7f6edfbebf)

## 📁 Estructura del Proyecto

```
.
├── Docs/                            #Documentos de investigación 
├── models/
│   └── tinyllama-gaming-1b-lora/    # Modelo fine-tuned con adaptadores LoRA
├── notebooks/                        # Jupyter notebooks para experimentación
├── utils/                           # Utilidades y funciones auxiliares
├── .gitattributes                   # Configuración de atributos de Git
├── .gitignore                       # Archivos ignorados por Git
├── README.md                        # Este archivo
├── app.py                          # Aplicación principal
└── requirements.txt                # Dependencias del proyecto
```

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

Ejecuta la aplicación principal:
```bash
python app.py
```

## 🔧 Características

- ✅ Fine-tuning con LoRA para eficiencia de memoria
- ✅ Especialización en contenido gaming con reviews
- ✅ Modelo base TinyLlama-1.1B
- ✅ Interfaz interactiva mediante Gradio para despliegue en HF Spaces

## 📊 Notebooks

Los notebooks incluidos permiten:
- Exploración de datos
- Proceso de fine-tuning
- Evaluación del modelo
- Experimentación con hiperparámetros

## 🛠️ Tecnologías Utilizadas

- Python
- Transformers (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- PyTorch

## 📝 Requisitos

Ver `requirements.txt` para la lista completa de dependencias.

## 👥 Autores

- Juan Camilo Niño
- Nicolás Acevedo
- Simón Porras Villalobos

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub