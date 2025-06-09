# Contenido ACTUALIZADO para lanistr/dataset/custom/load_custom.py

import os
import pandas as pd
import omegaconf
from sklearn.model_selection import train_test_split

from .custom_data_processor import DataProcessor
from .custom_dataset import CustomTextTabularDataset

def _split_data(df: pd.DataFrame):
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_val_df, test_size=1/9, random_state=42)
    print(f"División de datos: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    return train_df, valid_df, test_df

def load_custom_data(args: omegaconf.DictConfig, tokenizer):
    print("--- Iniciando Carga de Datos Personalizada ---")

    try:
        data_schema = args.data_schema
        print("Schema de datos cargado desde el archivo de configuración.")
    except omegaconf.errors.ConfigAttributeError:
        raise AttributeError("La configuración debe contener una sección 'data_schema' con las definiciones de las columnas.")

    file_path = os.path.join(args.data_dir, f"{args.category}.csv")
    print(f"Cargando datos desde: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no fue encontrado.")

    full_df = pd.read_csv(file_path, delimiter=';')

    train_df, valid_df, test_df = _split_data(full_df)

    processor_args = {**data_schema, 'tokenizer': tokenizer}
    processor = DataProcessor(**processor_args)

    train_processed = processor.process_data(train_df, fit=True)
    valid_processed = processor.process_data(valid_df, fit=False)
    test_processed = processor.process_data(test_df, fit=False)

    train_dataset = CustomTextTabularDataset(train_processed)
    valid_dataset = CustomTextTabularDataset(valid_processed)
    test_dataset = CustomTextTabularDataset(test_processed)

    # --- BLOQUE ACTUALIZADO ---
    # Calculamos los índices y dimensiones categóricas y los añadimos a la información tabular.
    num_numeric_features = len(data_schema['numeric_cols'])
    num_categorical_features = len(data_schema['categorical_cols'])
    
    cat_idxs = list(range(num_numeric_features, num_numeric_features + num_categorical_features))
    
    tabular_info = {
        'input_dim': train_processed['tabular_features'].shape[1],
        'feature_names': data_schema['numeric_cols'] + data_schema['categorical_cols'],
        'cat_idxs': cat_idxs,
        'cat_dims': processor.cat_dims, # Obtenemos las dimensiones guardadas por el procesador
    }
    # --------------------------

    print("--- Carga de Datos Personalizada Finalizada ---")

    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
        'tabular_data_information': tabular_info,
    }