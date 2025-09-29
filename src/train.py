# src/train.py
"""
Script de treinamento e seleção de modelos (Tech Challenge)

Objetivo
--------
Comparar múltiplos algoritmos de classificação (LogisticRegression,
DecisionTree, RandomForest e SVC) para prever se um tumor de mama é
maligno ou benigno (dataset público Breast Cancer Wisconsin - Diagnostic).

"""

# Bibliotecas padrão e utilitários
import json
from pathlib import Path

# Terceiros
import joblib
import numpy as np
import pandas as pd

# scikit-learn - dados, modelos, avaliação
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Parâmetros globais
RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "f1"

# 1) Garantir estrutura de pastas
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("models/experiments").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

# 2) Carregar dataset e salvar CSV bruto
data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("data/raw/wdbc.csv", index=False)  # reprodutibilidade

# 3) X (features) e y (alvo)
X = df.drop(columns=["target"])
y = df["target"]

# 4) Split estratificado (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 5) Definir pipelines por modelo
# Obs.: escalonamento é necessário para modelos baseados em distância/margem (LogReg/SVC).
model_pipelines = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "dtree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]),
    "rf": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "svc": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
}

# Validação cruzada (F1, 5-fold) para cada modelo
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_means, cv_stds = {}, {}

for name, pipe in model_pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    cv_means[name] = float(np.mean(scores))
    cv_stds[name]  = float(np.std(scores))

# 6) Salvar ranking da seleção de modelos
selection_df = (
    pd.DataFrame({
        "model": list(cv_means.keys()),
        "mean_f1_cv": list(cv_means.values()),
        "std_f1_cv": [cv_stds[m] for m in cv_means.keys()],
    })
    .sort_values(by="mean_f1_cv", ascending=False)
    .reset_index(drop=True)
)
selection_df.to_csv("reports/model_selection.csv", index=False)

# 7) Treinar CADA modelo no treino completo e salvar em experiments
#    (também vamos calcular métricas no TESTE para todos e salvar em relatório)
test_rows = []

trained_models = {}
for name, pipe in model_pipelines.items():
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe  # manter em memória

    # salvar pipeline treinado individualmente
    joblib.dump(pipe, f"models/experiments/{name}.joblib")

    # avaliação no teste (para relatório)
    y_pred = pipe.predict(X_test)
    # Para modelos com predict_proba; SVC tem probability=True no pipeline
    y_proba = pipe.predict_proba(X_test)[:, 1]

    row = {
        "model": name,
        "f1_test": float(f1_score(y_test, y_pred)),
        "accuracy_test": float(accuracy_score(y_test, y_pred)),
        "precision_test": float(precision_score(y_test, y_pred)),
        "recall_test": float(recall_score(y_test, y_pred)),
        "roc_auc_test": float(roc_auc_score(y_test, y_proba)),
    }
    test_rows.append(row)

test_metrics_df = pd.DataFrame(test_rows).sort_values(by="f1_test", ascending=False).reset_index(drop=True)
test_metrics_df.to_csv("reports/test_metrics.csv", index=False)

# 8) Escolher campeão (maior F1 médio na CV), avaliar no TESTE e salvar artefatos finais
best_name = selection_df.iloc[0]["model"]
best_model = trained_models[best_name]

# Avaliação final detalhada do campeão
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

best_metrics = {
    "f1": float(f1_score(y_test, y_pred_best)),
    "accuracy": float(accuracy_score(y_test, y_pred_best)),
    "precision": float(precision_score(y_test, y_pred_best)),
    "recall": float(recall_score(y_test, y_pred_best)),
    "roc_auc": float(roc_auc_score(y_test, y_proba_best)),
    "confusion_matrix": confusion_matrix(y_test, y_pred_best).tolist(),
}

# Salvar campeão para o app (consumido pelo deploy)
joblib.dump(best_model, "models/model.joblib")

# Metadados ricos (úteis para auditoria acadêmica)
metadata = {
    "best_model": best_name,
    "cv_means": cv_means,
    "cv_stds": cv_stds,
    "metrics": best_metrics,
    "feature_names": list(X.columns),
    "class_names": list(data.target_names),
    "random_state": RANDOM_STATE,
    "cv_scoring": SCORING,
    "cv_n_splits": N_SPLITS,
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Seleção concluída.")
print(f"   - Campeão: {best_name}")
print("   - Artefatos finais: models/model.joblib, models/metadata.json")
print("   - Relatórios: reports/model_selection.csv, reports/test_metrics.csv")
print("   - Experimentos: models/experiments/<modelo>.joblib")