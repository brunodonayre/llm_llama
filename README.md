# Fine-tuning en Español con LLaMA

Este repositorio contiene un script para realizar fine-tuning de un modelo LLaMA en español usando la librería `transformers` y `peft`. El entrenamiento se basa en técnicas de ajuste eficiente como LoRA y emplea `datasets` personalizadas.

## 📁 Archivos

- `11_3_finetune_llama_spanish.py`: Script principal para fine-tuning.
- `11_3_finetune_llama_spanish.ipynb`: Notebook original para exploración y ejecución paso a paso.

## 📦 Requisitos

Instala los siguientes paquetes para ejecutar el script:

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
