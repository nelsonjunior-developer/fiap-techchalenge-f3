# fiap-techchalenge-f3


Este projeto é um desafio acadêmico da FIAP, focado em Machine Learning aplicado a um problema de classificação. Utiliza um dataset público e envolve todas as etapas do ciclo de vida de um modelo de ML, desde a coleta e armazenamento dos dados até o deploy em nuvem por meio de uma aplicação Streamlit.

## Versão do Python

O projeto foi desenvolvido e testado com Python 3.11.9. Recomenda-se utilizar Python 3.11 ou superior para garantir total compatibilidade com as dependências e scripts fornecidos.

## Objetivo do Projeto e Visão Geral do Modelo de Machine Learning

O objetivo deste projeto é desenvolver um modelo de classificação binária que possa prever corretamente a classe alvo a partir de um conjunto de dados público e estruturado. O desafio envolve diversas etapas fundamentais do ciclo de vida de um projeto de Machine Learning:

- **Coleta e Armazenamento dos Dados:** Utilização de um dataset público, armazenado em formato CSV para facilitar o acesso e manipulação.
- **Análise Exploratória:** Realização de análises estatísticas e visuais para compreender as características dos dados, identificar padrões, detectar outliers e possíveis inconsistências.
- **Pré-processamento:** Limpeza, transformação e preparação dos dados para garantir que estejam em um formato adequado para o treinamento do modelo.
- **Modelagem:** Teste e avaliação de diferentes algoritmos de classificação para identificar o modelo que apresenta melhor desempenho para o problema proposto.
- **Deploy:** Implementação do modelo treinado em uma aplicação web usando Streamlit, permitindo a interação e utilização do modelo em ambiente de nuvem.

### Detalhes técnicos essenciais

- **Dataset e features:** Utilizamos o **Breast Cancer Wisconsin (Diagnostic – WDBC)** com **30 features numéricas** extraídas de imagens (medidas como raio, textura, concavidade, etc.).  
- **Variável alvo (`target`):** é a coluna que o modelo deve prever, com o mapeamento **`0 = malignant (maligno)`** e **`1 = benign (benigno)`**. As 30 demais colunas são as **features** usadas para a classificação.
- **Esquema de entrada (app):**
  - **Formulário:** campos para as 30 features numéricas.  
  - **Predição em lote (CSV):** o arquivo deve conter **as mesmas 30 colunas** do treino. A ordem das colunas pode variar; o app reordena internamente usando `feature_names` de `models/metadata.json`.
- **Reprodutibilidade:** usamos **`random_state = 42`** em todo o pipeline e **split estratificado (80/20)** para garantir comparabilidade e manter a proporção de classes.
- **Pré-processamento:** **padronização (StandardScaler)** aplicada **dentro do Pipeline** para os modelos que precisam de escala. Assim, o mesmo pré-processamento é aplicado de forma consistente no treino e na inferência.
- **Algoritmos comparados:** **LogisticRegression** (`class_weight="balanced"`), **DecisionTree**, **RandomForest** (`n_estimators=300`) e **SVC (RBF, probability=True, class_weight="balanced")**.
- **Seleção de modelo:** **validação cruzada estratificada (5-fold)** com métrica **F1**; selecionamos o modelo com **maior F1 médio**. Em seguida, refazemos o treino no conjunto de treino completo e avaliamos no **teste**.
- **Métricas reportadas:** **F1** (principal), **accuracy**, **precision**, **recall**, **ROC AUC** e **matriz de confusão**. Consideramos o leve desbalanceamento de classes via `class_weight="balanced"`.
- **Probabilidades no app:** exibimos `probabilidade_benign = P(benign)`. Além disso, o CSV de predição em lote inclui a coluna **`confiança_predicao`**, que mostra a **probabilidade da classe prevista** (se a classe prevista é benigna, usamos `P(benign)`; se for maligna, usamos `1 - P(benign)`). O **limiar padrão** é **0.5**.
- **Artefatos do modelo:**  
  - `models/model.joblib` → **Pipeline** completo (pré-processamento + classificador) pronto para `predict()` e `predict_proba()`.  
  - `models/metadata.json` → metadados (lista de `feature_names`, `class_names`, métricas do campeão, `cv_means`/`cv_stds`, `random_state`, etc.).
- **Deploy:** o app em Streamlit consome esses artefatos versionados no repositório e disponibiliza uma URL pública para uso e avaliação.

Este fluxo completo garante que o projeto não apenas desenvolva um modelo eficiente, mas também entregue uma solução funcional e acessível para o usuário final.

## Contexto do Problema

Este projeto aborda a detecção de câncer de mama, com foco em prever se um tumor é maligno ou benigno a partir de características obtidas em exames de imagem. Trata-se de um problema de classificação binária em Machine Learning, baseado no dataset público **Breast Cancer Wisconsin (Diagnostic)**.

A detecção precoce e precisa do câncer de mama é fundamental para aumentar as chances de sucesso no tratamento e reduzir a mortalidade. Utilizar modelos preditivos pode apoiar a triagem médica, auxiliando profissionais de saúde a tomar decisões mais rápidas e assertivas sobre o diagnóstico. Assim, o modelo desenvolvido neste projeto pode ser uma ferramenta importante no apoio à tomada de decisão clínica, proporcionando uma segunda opinião baseada em dados objetivos extraídos de exames.

## Seleção de Modelos e Relatórios

O script `src/train.py` realiza a comparação entre quatro algoritmos de classificação: LogisticRegression, DecisionTree, RandomForest e SVC. A avaliação dos modelos é feita utilizando validação cruzada (cross-validation) com 5 folds, tendo como métrica principal o F1-score.

Durante o processo de modelagem, são gerados e salvos os seguintes artefatos e relatórios:

- O modelo campeão (com melhor desempenho na validação cruzada) é salvo em `models/model.joblib`.
- Metadados do processo de seleção são salvos em `models/metadata.json`, incluindo métricas de validação, scores de cross-validation e informações sobre o modelo campeão.
- Todos os modelos treinados (incluindo os que não foram escolhidos como campeão) são salvos individualmente em `models/experiments/`.
- O ranking de desempenho dos modelos (baseado nos resultados da validação cruzada) é registrado em `reports/model_selection.csv`.
- As métricas de teste de todos os modelos avaliados são documentadas em `reports/test_metrics.csv`.

Esses relatórios e artefatos garantem transparência acadêmica no processo de desenvolvimento, permitindo a auditoria das escolhas de modelo e facilitando a reprodutibilidade dos resultados.

## Estrutura do Projeto
```
tech-challenge/
│
├── data/
│   ├── raw/                 # Dados originais (CSV exportado do dataset)
│
├── notebooks/
│   └── eda.ipynb            # Análises exploratórias e estatísticas
│
├── src/
│   ├── train.py             # Treino, comparação (CV), artefatos e relatórios
│   └── evaluate.py          # Avaliação visual (CM/ROC/PR) e comparativos
│
├── models/
│   ├── model.joblib         # Modelo campeão (pipeline completo)
│   ├── metadata.json        # Métricas, CV e metadados
│   └── experiments/         # Modelos treinados de cada algoritmo
│
├── reports/
│   ├── model_selection.csv  # Ranking (CV)
│   ├── test_metrics.csv     # Métricas em teste por modelo
│   └── figures/             # Gráficos (CM, ROC, PR, comparativos)
│
├── app.py                   # Aplicação Streamlit (deploy em nuvem)
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação principal
└── LICENSE                  # Licença do projeto
```

## Avaliação e Relatórios

O script `src/evaluate.py` gera gráficos e relatórios adicionais para complementar a análise do modelo campeão e a comparação entre modelos.

Ele cria e salva:

- Matriz de confusão, curva ROC e curva Precision-Recall do modelo campeão (`reports/figures/`).
- Classification report em texto (`reports/classification_report.txt`).
- Gráficos de barras comparativos entre modelos usando os relatórios de seleção (`reports/model_selection.csv` e `reports/test_metrics.csv`).

Esses outputs enriquecem a análise, facilitam a interpretação dos resultados.

## Checklist de Requisitos (Tech Challenge)

- [X] Definição do problema (Classificação binária)
- [X] Coleta/armazenamento dos dados (dataset público salvo em CSV)
- [X] Análise exploratória (EDA notebook)
- [X] Processamento dos dados (split, escalonamento, balanceamento via class_weight)
- [X] Modelagem (comparação de 4 algoritmos, seleção por F1)
- [X] Avaliação (métricas, gráficos, relatórios)
- [X] Deploy (aplicação Streamlit em nuvem)

## Deploy da Aplicação


A aplicação está disponível publicamente via Streamlit Cloud: [https://fiap-techchallenge-f3.streamlit.app](https://fiap-techchallenge-f3.streamlit.app)

## Como Rodar o Notebook

Para executar o notebook exploratório (`notebooks/eda.ipynb`) localmente, siga os passos abaixo:

1. **Ative o ambiente virtual**:
   ```
   source .venv/bin/activate
   ```

2. **Instale as dependências**:
   ```
   pip install -r requirements.txt
   ```

3. **Garanta que o pacote `ipykernel` está instalado**  
   (já incluído no `requirements.txt`).

4. **(Opcional) Registre o ambiente virtual como kernel do Jupyter**  
   Caso deseje selecionar o kernel correspondente no Jupyter:
   ```
   python -m ipykernel install --user --name=tech-challenge-env --display-name "Python (tech-challenge)"
   ```

5. **Abra o notebook**:
   ```
   jupyter notebook notebooks/eda.ipynb
   ```

6. **No Jupyter, selecione o kernel correto**  
   Escolha o kernel chamado **Python (tech-challenge)** para garantir que o ambiente e as dependências corretas serão utilizados.