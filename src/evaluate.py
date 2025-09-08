# src/evaluate.py
"""
Módulo de avaliação e visualização dos resultados (Tech Challenge)

Objetivo
--------
Gerar visualizações e relatórios do modelo campeão e da comparação entre modelos:
1) Carregar artefatos finais (models/model.joblib, models/metadata.json).
2) Plotar e salvar:
   - Matriz de confusão
   - Curva ROC
   - Curva Precision-Recall (PR)
3) Ler relatórios da seleção (reports/model_selection.csv, reports/test_metrics.csv)
   e gerar gráficos comparativos de desempenho entre modelos.
4) Exportar figuras para reports/figures/ e classification_report.txt para documentação.

Pré-requisitos
--------------
- Executar antes o treinamento/seleção:
  `python src/train.py`

Saídas principais
-----------------
- reports/figures/confusion_matrix.png
- reports/figures/roc_curve.png
- reports/figures/precision_recall_curve.png
- reports/figures/cv_mean_f1_by_model.png
- reports/figures/test_f1_by_model.png
- reports/classification_report.txt
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

# -----------------------------------------------------------------------------
# Configurações globais
# -----------------------------------------------------------------------------
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = Path("models/metadata.json")
MODEL_PATH = Path("models/model.joblib")
SELECTION_CSV = Path("reports/model_selection.csv")
TEST_METRICS_CSV = Path("reports/test_metrics.csv")

# -----------------------------------------------------------------------------
# Utilitários de plot
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    """Plota e salva matriz de confusão simples (matplotlib puro)."""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Anotações nas células
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_score, out_path: Path):
    """Plota e salva a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(y_true, y_score, out_path: Path):
    """Plota e salva a curva Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def barh_scores(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    """Gráfico de barras horizontais para comparação de modelos."""
    fig = plt.figure(figsize=(6, 4))
    y = df[y_col]
    x = df[x_col]
    plt.barh(y, x)
    plt.xlabel(x_col.replace("_", " ").upper())
    plt.ylabel(y_col.replace("_", " ").upper())
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Função principal
# -----------------------------------------------------------------------------
def main():
    # 1) Carregar metadados + modelo campeão
    if not META_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Artefatos não encontrados. Rode `python src/train.py` antes de avaliar."
        )

    with open(META_PATH) as f:
        meta = json.load(f)

    model = joblib.load(MODEL_PATH)
    feature_names = meta["feature_names"]
    class_names = meta["class_names"]
    best_metrics = meta["metrics"]

    # 2) Para gerar as curvas/CM precisamos dos dados de teste
    #    A estratégia aqui: recarregar o CSV bruto e refazer o split com o mesmo random_state
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=meta.get("random_state", 42)
    )

    # Predições/probabilidades do campeão
    y_pred = model.predict(X_test)
    # Probabilidade da classe positiva (convenção do treino: índice 1 = 'benign')
    y_score = model.predict_proba(X_test)[:, 1]

    # 3) Gerar e salvar gráficos do campeão
    plot_confusion_matrix(
        y_test, y_pred, labels=["malignant", "benign"],
        out_path=FIG_DIR / "confusion_matrix.png"
    )
    plot_roc(y_test, y_score, out_path=FIG_DIR / "roc_curve.png")
    plot_precision_recall(y_test, y_score, out_path=FIG_DIR / "precision_recall_curve.png")

    # 4) Salvar classification_report em texto (útil para anexar ao relatório)
    report_txt = classification_report(
        y_test, y_pred, target_names=["malignant", "benign"]
    )
    (Path("reports") / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # 5) Comparação entre modelos — CV e Teste
    # CV (ranking de f1 na validação cruzada)
    if SELECTION_CSV.exists():
        sel_df = pd.read_csv(SELECTION_CSV)
        # Espera colunas: model, mean_f1_cv, std_f1_cv
        # Ordena do melhor para pior (se já não estiver)
        sel_df = sel_df.sort_values(by="mean_f1_cv", ascending=True)  # ascending=True para barh (de baixo p/ cima)
        barh_scores(
            df=sel_df,
            x_col="mean_f1_cv",
            y_col="model",
            title="CV (F1 médio, 5-fold) por modelo",
            out_path=FIG_DIR / "cv_mean_f1_by_model.png",
        )

    # Teste (F1 final em teste por modelo)
    if TEST_METRICS_CSV.exists():
        test_df = pd.read_csv(TEST_METRICS_CSV)
        # Espera colunas: model, f1_test, accuracy_test, precision_test, recall_test, roc_auc_test
        test_df = test_df.sort_values(by="f1_test", ascending=True)
        barh_scores(
            df=test_df,
            x_col="f1_test",
            y_col="model",
            title="F1 em teste por modelo",
            out_path=FIG_DIR / "test_f1_by_model.png",
        )

    # 6) Resumo no console (útil para logs)
    print("✅ Avaliação concluída. Figuras em reports/figures/")
    print(f"   - Best model: {meta.get('best_model')}")
    print(f"   - Métricas do campeão (teste): {best_metrics}")
    print("   - Relatórios lidos (se existentes):")
    print(f"     * {SELECTION_CSV}  -> ranking CV")
    print(f"     * {TEST_METRICS_CSV} -> métricas de teste por modelo")


if __name__ == "__main__":
    main()