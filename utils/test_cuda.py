#import torch
#print("CUDA disponible:", torch.cuda.is_available())
#print(torch.version.cuda)
#print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

#torch.cuda.empty_cache()

from huggingface_hub import snapshot_download
import shutil, os

model_cache_dir = snapshot_download("mistralai/Mistral-7B-Instruct-v0.3")
shutil.rmtree(model_cache_dir)  # borra la carpeta del modelo
