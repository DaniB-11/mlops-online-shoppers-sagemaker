import os
import pandas as pd
import xgboost as xgb
import json
import tarfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

if __name__ == "__main__":
    print("### Evaluando modelo...")

    # Descomprimir modelo
    tar_path = "/opt/ml/processing/model/model.tar.gz"
    extracted_path = "/opt/ml/processing/model/"
    
    if tarfile.is_tarfile(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extracted_path)

    model_path = os.path.join(extracted_path, "xgboost-model.model")
    if not os.path.exists(model_path):
        print("Modelo no encontrado.")
        exit(1)

    # Cargar test
    test_x = pd.read_csv("/opt/ml/processing/test/test_script_x.csv", header=None)
    test_y = pd.read_csv("/opt/ml/processing/test/test_script_y.csv", header=None)

    model = xgb.Booster()
    model.load_model(model_path)

    dtest = xgb.DMatrix(test_x)
    preds_prob = model.predict(dtest)
    preds = [1 if p > 0.5 else 0 for p in preds_prob]

    # Métricas
    accuracy = accuracy_score(test_y, preds)
    precision = precision_score(test_y, preds)
    recall = recall_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    cm = confusion_matrix(test_y, preds)

    print("### Resultados:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar métricas
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, f)

    # Guardar matriz como CSV
    cm_df = pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"], columns=["Pred Neg", "Pred Pos"])
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    print("### Evaluación completada y matriz de confusión guardada ✅")