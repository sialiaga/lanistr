# Contenido FINAL Y CORREGIDO para lanistr/dataset/custom/custom_dataset.py

import torch
from torch.utils.data import Dataset

class CustomTextTabularDataset(Dataset):
    """
    Un Dataset de PyTorch limpio y eficiente para datos de texto y tabulares
    que ya han sido pre-procesados.
    """
    def __init__(self, processed_data: dict):
        """
        Se inicializa directamente con un diccionario de tensores ya procesados.
        """
        self.input_ids = processed_data['input_ids']
        self.attention_mask = processed_data['attention_mask']
        self.tabular_features = processed_data['tabular_features']
        self.labels = processed_data['labels']

    def __len__(self):
        """Devuelve la cantidad total de muestras en el dataset."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Devuelve una única muestra del dataset.
        Añadimos una dimensión extra a los tensores de texto para que coincidan
        con la forma 3D que espera la arquitectura de Lanistr.
        """
        return {
            # --- LÍNEAS CORREGIDAS ---
            # .unsqueeze(0) añade la dimensión de "num_campos_de_texto" (que es 1).
            'input_ids': self.input_ids[index].unsqueeze(0),
            'attention_mask': self.attention_mask[index].unsqueeze(0),
            # -------------------------
            'features': self.tabular_features[index], 
            'labels': self.labels[index]
        }