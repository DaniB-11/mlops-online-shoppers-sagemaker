import pandas as pd
import numpy as np
import argparse
import os

# Instalar imblearn si no está disponible
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imblearn"])
    from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =============================
#  Argumentos del Script
# =============================
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--train_output', type=str, default='/opt/ml/processing/output/train')
    parser.add_argument('--validation_output', type=str, default='/opt/ml/processing/output/validation')
    parser.add_argument('--test_output', type=str, default='/opt/ml/processing/output/test')
    parser.add_argument('--filename', type=str, default='online_shoppers_intention.csv')
    return parser.parse_args()

# =============================
#  Carga de Datos
# =============================
def load_data(filepath, filename):
    file_path = os.path.join(filepath, filename)
    df = pd.read_csv(file_path)
    return df

# =============================
#  Preprocesamiento General
# =============================
def preprocess_and_split(df):
    y = df['Revenue'].astype(int)
    X = df.drop('Revenue', axis=1)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_columns = np.concatenate([num_cols, cat_features])

    X_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
                        columns=all_columns)

    df_processed = pd.concat([X_df, pd.Series(y.values, name='Revenue')], axis=1)
    df_processed = df_processed.sample(frac=1, random_state=42)  # shuffle

    train, val, test = np.split(df_processed, [int(0.7 * len(df_processed)), int(0.9 * len(df_processed))])
    return train, val, test

# =============================
# Aplicar SMOTE
# =============================
def apply_smote(train_df):
    X_train = train_df.drop('Revenue', axis=1)
    y_train = train_df['Revenue']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    df_resampled = pd.concat([pd.Series(y_res, name='Revenue'), pd.DataFrame(X_res, columns=X_train.columns)], axis=1)
    return df_resampled

# =============================
# Guardar archivos procesados
# =============================
def save_data(train_data, val_data, test_data, train_output, val_output, test_output):
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # Asegurar que Revenue esté como primera columna en todos los sets
    train_data = train_data[['Revenue'] + [col for col in train_data.columns if col != 'Revenue']]
    val_data = val_data[['Revenue'] + [col for col in val_data.columns if col != 'Revenue']]
    test_data = test_data[['Revenue'] + [col for col in test_data.columns if col != 'Revenue']]

    # Guardar datos
    train_data.to_csv(os.path.join(train_output, 'train_script.csv'), index=False, header=False)
    val_data.to_csv(os.path.join(val_output, 'validation_script.csv'), index=False, header=False)

    test_data['Revenue'].to_csv(os.path.join(test_output, 'test_script_y.csv'), index=False, header=False)
    test_data.drop(['Revenue'], axis=1).to_csv(os.path.join(test_output, 'test_script_x.csv'), index=False, header=False)

# =============================
# Ejecutar procesamiento
# =============================
def main():
    args = _parse_args()

    print("### Cargando datos...")
    df = load_data(args.input, args.filename)

    print("### Preprocesando y dividiendo...")
    train_df, val_df, test_df = preprocess_and_split(df)

    print("### Aplicando SMOTE al set de entrenamiento...")
    train_df_smote = apply_smote(train_df)

    print("### Guardando datos preprocesados...")
    save_data(train_df_smote, val_df, test_df, args.train_output, args.validation_output, args.test_output)

    print("### ✅ Procesamiento completado.")

if __name__ == '__main__':
    main()