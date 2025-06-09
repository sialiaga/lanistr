# Contenido para lanistr/dataset/custom/custom_dataset.py

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
        
        Args:
            processed_data (dict): Un diccionario que debe contener 'input_ids', 
                                   'attention_mask', 'tabular_features' y 'labels' como tensores.
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
        Es una operación muy rápida de indexación de tensores.
        """
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            # Nombramos la clave 'features' para mantener consistencia con el código original de Lanistr
            'features': self.tabular_features[index], 
            'labels': self.labels[index]
        }