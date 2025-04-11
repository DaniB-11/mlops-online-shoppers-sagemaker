import xgboost as xgb
import argparse
import os
import pandas as pd
import sys

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return xgb.DMatrix(X, label=y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argumentos obligatorios de SageMaker
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    # Argumentos de hiperparámetros (para tuning)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=1.0)

    args = parser.parse_args()

    try:
        print("### Cargando datos...")
        dtrain = load_data(os.path.join(args.train, 'train_script.csv'))
        dvalid = load_data(os.path.join(args.validation, 'validation_script.csv'))

        # Usar hiperparámetros pasados como argumentos
        params = {
            'max_depth': args.max_depth,
            'eta': args.eta,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss', 'error']
        }

        print("### Hiperparámetros usados:")
        print(params)

        print("### Entrenando modelo...")
        model = xgb.train(
            params,
            dtrain,
            evals=[(dvalid, 'validation')],
            num_boost_round=100,
            early_stopping_rounds=10
        )

        print("### Guardando modelo entrenado...")
        model_output_path = os.path.join(os.environ.get('SM_MODEL_DIR'), 'xgboost-model.model')
        model.save_model(model_output_path)

        print(f"### Modelo guardado en: {model_output_path}")
        print("Contenido del directorio del modelo:")
        print(os.listdir(os.environ.get('SM_MODEL_DIR')))

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}", file=sys.stderr)
        sys.exit(1)# Modelo guardado en: {model_output_path}")