# app.py
"""
Aplicação Streamlit para servir o modelo de classificação (Tech Challenge)

Objetivo
--------
Este app web coloca o modelo treinado em produção de forma simples:
1) Carrega os artefatos do modelo (pipeline .joblib) e metadados (features, classes, métricas).
2) Disponibiliza uma interface para:
   - Predizer a classe (maligno/benigno) a partir de um formulário com as features.
   - Fazer predições em lote a partir de um CSV com as mesmas colunas de treino.
   - Visualizar as métricas de avaliação do modelo (F1, Accuracy, Precision, Recall, ROC AUC).
3) Entrega uma URL pública quando hospedado em Streamlit Cloud/Spaces, atendendo ao requisito
   de "modelo produtivo (aplicação simples)".

Pré-requisitos
--------------
- Executar antes: `python src/train.py` para gerar:
  - models/model.joblib
  - models/metadata.json
- Ter `requirements.txt` com as dependências (streamlit, scikit-learn, pandas, joblib).

Como usar localmente
--------------------
- streamlit run app.py
- (para deploy, basta apontar o provedor para este arquivo)
"""

# Bibliotecas padrão
import json
from pathlib import Path

# Bibliotecas de terceiros
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuração geral da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Detecção de Câncer de Mama (WDBC)",
    layout="wide",
    page_icon="🩺"
)

# -----------------------------------------------------------------------------
# Funções utilitárias
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Carrega o modelo treinado (.joblib) e o arquivo de metadados (JSON).

    Retorna
    -------
    model: objeto sklearn Pipeline
        Pipeline com pré-processamento + classificador treinado.
    meta: dict
        Dicionário com 'feature_names', 'class_names' e 'metrics'.
    """
    model_path = Path("models/model.joblib")
    meta_path = Path("models/metadata.json")

    if not model_path.exists() or not meta_path.exists():
        # Mensagem amigável se o usuário esqueceu de treinar o modelo antes do deploy
        st.error(
            "Artefatos do modelo não encontrados.\n\n"
            "Certifique-se de executar `python src/train.py` antes do deploy "
            "para gerar `models/model.joblib` e `models/metadata.json`."
        )
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """
    Verifica se o DataFrame possui todas as colunas necessárias.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dados de entrada.
    required_cols : list[str]
        Lista de nomes de colunas esperadas (na ordem do treino).

    Retorna
    -------
    missing : list[str]
        Lista de colunas faltantes.
    """
    return [c for c in required_cols if c not in df.columns]


# -----------------------------------------------------------------------------
# Carregamento dos artefatos
# -----------------------------------------------------------------------------
model, meta = load_artifacts()
feature_names: list[str] = meta["feature_names"]
class_names: list[str] = meta["class_names"]
metrics: dict = meta.get("metrics", {})

# -----------------------------------------------------------------------------
# Layout principal
# -----------------------------------------------------------------------------
st.title("🩺 Classificação de Tumores de Mama — WDBC")
st.caption(
    "Aplicação web para inferência do modelo treinado no dataset "
    "Breast Cancer Wisconsin (Diagnostic)."
)

# Cria abas para organizar a navegação no app
tab_form, tab_batch, tab_metrics = st.tabs(
    ["Predizer (formulário)", "Predizer em lote (CSV)", "Métricas do modelo"]
)

# -----------------------------------------------------------------------------
# Aba 1: Predição por formulário
# -----------------------------------------------------------------------------
with tab_form:
    st.subheader("Entrada manual das features")
    st.write(
        "Preencha as **30 features numéricas** abaixo. "
        "Se você não souber os valores reais, pode testar com valores padrão."
    )

    # Organizamos os inputs em colunas para melhor usabilidade
    cols = st.columns(3)
    form_values = {}

    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            # number_input com valor inicial 0.0 e 4 casas decimais
            form_values[feat] = st.number_input(feat, value=0.0, step=0.0001, format="%.4f")

    # Botão de predição
    if st.button("Predizer (formulário)"):
        # Constrói um DataFrame de uma linha com a ordem correta de colunas
        X = pd.DataFrame([form_values])[feature_names]

        # Predição de classe e probabilidade da classe positiva (convenção: 'benign')
        proba_benign = float(model.predict_proba(X)[0, 1])
        pred_class_idx = int(model.predict(X)[0])
        pred_class = class_names[pred_class_idx]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Classe prevista", pred_class)
        with c2:
            st.metric("Probabilidade de 'benign'", f"{proba_benign:.3f}")

        # Observação clara sobre interpretação do score
        st.info(
            "A probabilidade exibida corresponde à classe **'benign'**. "
            "Valores próximos de 1 indicam maior confiança de benignidade; "
            "próximos de 0 indicam maior confiança de malignidade."
        )

# -----------------------------------------------------------------------------
# Aba 2: Predição em lote (CSV)
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload de CSV para predição em lote")
    st.write(
        "Envie um **CSV** contendo **exatamente as mesmas colunas** usadas no treino "
        "(em qualquer ordem). O app validará os nomes e ordem internamente."
    )

    uploaded = st.file_uploader("Selecione um arquivo .csv", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.stop()

        # Validação de colunas
        missing = validate_columns(df_in, feature_names)
        if missing:
            st.error(f"Colunas faltantes no CSV: {missing}")
        else:
            # Gera as predições respeitando a ordem das features usada no treino
            Xb = df_in[feature_names]
            proba_benign = model.predict_proba(Xb)[:, 1]
            pred_idx = model.predict(Xb)

            # Monta DataFrame de saída com predições
            df_out = df_in.copy()
            df_out["pred_class"] = [class_names[i] for i in pred_idx]
            df_out["proba_benign"] = proba_benign

            st.success("Predições geradas com sucesso!")
            st.dataframe(df_out.head(), use_container_width=True)

            # Oferece download do resultado completo
            st.download_button(
                label="⬇️ Baixar predições (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="predicoes.csv",
                mime="text/csv",
            )

# -----------------------------------------------------------------------------
# Aba 3: Métricas do modelo
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Desempenho no conjunto de teste")
    if not metrics:
        st.warning(
            "Métricas não encontradas em `models/metadata.json`. "
            "Execute novamente `python src/train.py` para gerar."
        )
    else:
        # Exibe métricas principais em cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1", f"{metrics.get('f1', 0):.3f}")
        c2.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        c3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        c4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        st.caption(f"ROC AUC: {metrics.get('roc_auc', 0):.3f}")

        # Matriz de confusão (opcional: mostrada como tabela)
        cmat = metrics.get("confusion_matrix")
        if cmat:
            st.write("Matriz de confusão (linhas = verdadeiro, colunas = previsto):")
            st.dataframe(
                pd.DataFrame(cmat, index=["malignant", "benign"], columns=["malignant", "benign"]),
                use_container_width=True,
            )

# Rodapé com lembretes úteis
st.divider()
st.caption(
    " Dica: se você atualizar o repositório no GitHub, o Streamlit Cloud "
    "rebuilda a aplicação automaticamente com a nova versão."
)