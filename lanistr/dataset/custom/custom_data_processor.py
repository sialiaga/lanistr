# Contenido ACTUALIZADO para lanistr/dataset/custom/custom_data_processor.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, text_cols: list, numeric_cols: list, categorical_cols: list, target_col: str, *, tokenizer):
        self.text_cols = text_cols
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.tokenizer = tokenizer
        self.scaler = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}
        self.is_fitted = False
        # --- NUEVA LÍNEA ---
        # Guardaremos las dimensiones de las categorías aquí
        self.cat_dims = []

    def process_data(self, df: pd.DataFrame, fit: bool = False):
        print(f"🚀 Procesando DataFrame... (Modo Fit: {fit})")
        df = df.copy()

        print(f"✍️  Uniendo y procesando {len(self.text_cols)} columnas de texto...")
        text_series = df[self.text_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)

        text_encodings = self.tokenizer(
            text_series.tolist(), truncation=True, padding='max_length',
            max_length=512, return_tensors='pt'
        )

        print(f"🔢 Procesando {len(self.numeric_cols)} columnas numéricas y {len(self.categorical_cols)} categóricas...")

        for col in self.numeric_cols:
            if fit:
                mean_val = df[col].mean()
                setattr(self, f"{col}_mean", mean_val) # Guardamos la media para usarla después
                df[col] = df[col].fillna(mean_val)
            else:
                mean_val = getattr(self, f"{col}_mean", df[col].mean()) # Usamos la media guardada
                df[col] = df[col].fillna(mean_val)
        
        # Solo procesamos columnas numéricas si la lista no está vacía
        numeric_features = self.scaler.fit_transform(df[self.numeric_cols]) if fit else self.scaler.transform(df[self.numeric_cols]) if self.numeric_cols else np.array([[] for _ in range(len(df))])

        if self.categorical_cols:
            if fit:
                self.cat_dims = [] # Reseteamos las dimensiones en cada ajuste

            categorical_features_list = []
            for col in self.categorical_cols:
                df[col] = df[col].fillna('__MISSING__')
                if fit:
                    encoded = self.label_encoders[col].fit_transform(df[col])
                    # --- NUEVA LÍNEA ---
                    # Guardamos el número de clases únicas para esta columna
                    self.cat_dims.append(len(self.label_encoders[col].classes_))
                else:
                    classes = self.label_encoders[col].classes_
                    unknown_label = '__UNKNOWN__'
                    if unknown_label not in classes:
                        self.label_encoders[col].classes_ = np.append(classes, unknown_label)
                    df[col] = df[col].apply(lambda x: x if x in classes else unknown_label)
                    encoded = self.label_encoders[col].transform(df[col])
                categorical_features_list.append(encoded.reshape(-1, 1))

            categorical_features = np.hstack(categorical_features_list)
            tabular_features = np.hstack([numeric_features, categorical_features])
        else:
            tabular_features = numeric_features

        if fit:
            self.is_fitted = True

        labels = torch.tensor(df[self.target_col].values, dtype=torch.long)
        
        print("✅ ¡Procesamiento completado!")
        return {
            'input_ids': text_encodings['input_ids'],
            'attention_mask': text_encodings['attention_mask'],
            'tabular_features': torch.tensor(tabular_features, dtype=torch.float32),
            'labels': labels
        }