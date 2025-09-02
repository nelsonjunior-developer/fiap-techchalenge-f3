# fiap-techchalenge-f3

Este projeto é um desafio acadêmico da FIAP, focado em Machine Learning aplicado a um problema de classificação. Utiliza um dataset público e envolve todas as etapas do ciclo de vida de um modelo de ML, desde a coleta e armazenamento dos dados até o deploy em nuvem por meio de uma aplicação Streamlit.

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

```
tech-challenge/
│
├── data/
│   ├── raw/               # Dados originais (CSV exportado do dataset)
│   ├── processed/         # Dados tratados e divididos em treino/teste
│
├── notebooks/
│   └── eda.ipynb          # Análises exploratórias e estatísticas
│
├── src/
│   ├── train.py           # Treino do modelo, avaliação e salvamento de artefatos
│   ├── evaluate.py        # Scripts auxiliares de avaliação e gráficos
│   └── utils/             # Funções auxiliares de pré-processamento
│
├── models/
│   ├── model.joblib       # Modelo treinado
│   ├── scaler.joblib      # Escalonador salvo (se usado)
│   └── metadata.json      # Métricas e informações do treino
│
├── app.py                 # Aplicação Streamlit (deploy em nuvem)
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação principal
└── LICENSE                # Licença do projeto
```