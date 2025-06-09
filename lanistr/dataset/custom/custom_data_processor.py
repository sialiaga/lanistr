# Contenido para lanistr/dataset/custom/custom_data_processor.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertTokenizer

class DataProcessor:
    """
    Clase autocontenida y flexible para cargar y preprocesar datos
    tabulares (numéricos y categóricos) y de texto.
    """
    def __init__(self, text_cols: list, numeric_cols: list, categorical_cols: list, target_col: str, tokenizer_name: str = 'bert-base-uncased'):
        """
        Inicializa el procesador de datos con la configuración específica del dataset.

        Args:
            text_cols (list): Columnas a tratar como texto.
            numeric_cols (list): Columnas a tratar como numéricas.
            categorical_cols (list): Columnas a tratar como categóricas.
            target_col (str): La columna que contiene la etiqueta o valor a predecir.
            tokenizer_name (str): El identificador del tokenizador de Hugging Face a usar.
        """
        self.text_cols = text_cols
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col

        # --- Inicialización de Herramientas de Procesamiento ---
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}
        
        self.is_fitted = False # Control para ajustar los encoders/scalers solo una vez (con datos de entrenamiento)

    def process_data(self, df: pd.DataFrame, fit: bool = False):
        """
        Procesa un DataFrame completo. Separa la lógica de ajuste (fit) de la de transformación (transform).
        
        Args:
            df (pd.DataFrame): El DataFrame a procesar.
            fit (bool): Si es True, ajusta los scalers y encoders. Debe ser True para el set de entrenamiento.
        """
        print(f"🚀 Procesando DataFrame... (Modo Fit: {fit})")
        df = df.copy() # Evitar SettingWithCopyWarning
        
        # --- 1. Procesar Texto ---
        print("✍️  Procesando columnas de texto...")
        # Usamos la primera columna de texto definida. Se puede adaptar para concatenar varias.
        text_series = df[self.text_cols[0]].astype(str).fillna('') if self.text_cols else pd.Series([''] * len(df))
        
        text_encodings = self.tokenizer(
            text_series.tolist(),
            truncation=True,
            padding='max_length',
            max_length=128,      # Este valor puede necesitar ser ajustado según tus datos
            return_tensors='pt'
        )

        # --- 2. Procesar Datos Tabulares ---
        print("🔢 Procesando columnas tabulares...")
        
        # Imputación de valores faltantes
        for col in self.numeric_cols:
            if fit:
                mean_val = df[col].mean()
                self.scaler_mean = mean_val
                df[col] = df[col].fillna(mean_val)
            else:
                df[col] = df[col].fillna(self.scaler_mean)

        for col in self.categorical_cols:
            df[col] = df[col].fillna('__MISSING__') # Etiqueta estándar para valores categóricos faltantes

        # Procesamiento de columnas numéricas
        numeric_features = self.scaler.fit_transform(df[self.numeric_cols]) if fit else self.scaler.transform(df[self.numeric_cols])
        
        # Procesamiento de columnas categóricas
        categorical_features_list = []
        for col in self.categorical_cols:
            # Manejar etiquetas no vistas en los sets de validación/prueba
            if fit:
                encoded = self.label_encoders[col].fit_transform(df[col])
            else:
                # Reemplazar etiquetas desconocidas con un valor para "desconocido"
                classes = self.label_encoders[col].classes_
                unknown_label = '__UNKNOWN__'
                if unknown_label not in classes:
                    # Añadir la etiqueta de desconocido si no existe
                    self.label_encoders[col].classes_ = np.append(classes, unknown_label)
                
                df[col] = df[col].apply(lambda x: x if x in classes else unknown_label)
                encoded = self.label_encoders[col].transform(df[col])

            categorical_features_list.append(encoded.reshape(-1, 1))
        
        categorical_features = np.hstack(categorical_features_list)
        
        # Combinar todas las características tabulares
        tabular_features = np.hstack([numeric_features, categorical_features])
        
        if fit:
            self.is_fitted = True

        # --- 3. Extraer Etiquetas ---
        labels = torch.tensor(df[self.target_col].values, dtype=torch.long)
        
        print("✅ ¡Procesamiento completado!")

        return {
            'input_ids': text_encodings['input_ids'],
            'attention_mask': text_encodings['attention_mask'],
            'tabular_features': torch.tensor(tabular_features, dtype=torch.float32),
            'labels': labels
        }