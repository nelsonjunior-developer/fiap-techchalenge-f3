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


Este fluxo completo garante que o projeto não apenas desenvolva um modelo eficiente, mas também entregue uma solução funcional e acessível para o usuário final.

## Contexto do Problema

Este projeto aborda a detecção de câncer de mama, com foco em prever se um tumor é maligno ou benigno a partir de características obtidas em exames de imagem. Trata-se de um problema de classificação binária em Machine Learning, baseado no dataset público **Breast Cancer Wisconsin (Diagnostic)**.

A detecção precoce e precisa do câncer de mama é fundamental para aumentar as chances de sucesso no tratamento e reduzir a mortalidade. Utilizar modelos preditivos pode apoiar a triagem médica, auxiliando profissionais de saúde a tomar decisões mais rápidas e assertivas sobre o diagnóstico. Assim, o modelo desenvolvido neste projeto pode ser uma ferramenta importante no apoio à tomada de decisão clínica, proporcionando uma segunda opinião baseada em dados objetivos extraídos de exames.

## Estrutura do projeto

## Seleção de Modelos e Relatórios

O script `src/train.py` realiza a comparação entre quatro algoritmos de classificação: LogisticRegression, DecisionTree, RandomForest e SVC. A avaliação dos modelos é feita utilizando validação cruzada (cross-validation) com 5 folds, tendo como métrica principal o F1-score.

Durante o processo de modelagem, são gerados e salvos os seguintes artefatos e relatórios:

- O modelo campeão (com melhor desempenho na validação cruzada) é salvo em `models/model.joblib`.
- Metadados do processo de seleção são salvos em `models/metadata.json`, incluindo métricas de validação, scores de cross-validation e informações sobre o modelo campeão.
- Todos os modelos treinados (incluindo os que não foram escolhidos como campeão) são salvos individualmente em `models/experiments/`.
- O ranking de desempenho dos modelos (baseado nos resultados da validação cruzada) é registrado em `reports/model_selection.csv`.
- As métricas de teste de todos os modelos avaliados são documentadas em `reports/test_metrics.csv`.

Esses relatórios e artefatos garantem transparência acadêmica no processo de desenvolvimento, permitindo a auditoria das escolhas de modelo e facilitando a reprodutibilidade dos resultados.

```
tech-challenge/
│
├── data/
│   ├── raw/                 # Dados originais (CSV exportado do dataset)
│   └── processed/           # (opcional) Dados tratados/splits, se decidirmos salvar
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

- [ ] Definição do problema (Classificação binária)
- [ ] Coleta/armazenamento dos dados (dataset público salvo em CSV)
- [ ] Análise exploratória (EDA notebook)
- [ ] Processamento dos dados (split, escalonamento, balanceamento via class_weight)
- [ ] Modelagem (comparação de 4 algoritmos, seleção por F1)
- [ ] Avaliação (métricas, gráficos, relatórios)
- [ ] Deploy (aplicação Streamlit em nuvem)