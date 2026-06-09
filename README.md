# MVP Credit Card Churn — Parte 02 (Machine Learning & Analytics)

MVP individual da disciplina de Machine Learning & Analytics (Pós-Graduação em Ciência de
Dados e Analytics, PUC-Rio). Continuação do
[mvp-credit-card-churn-parte-01](https://github.com/diogorblessa/mvp-credit-card-churn-parte-01),
que cobriu a análise exploratória e o pré-processamento. Esta parte foca na **modelagem,
otimização e avaliação** de modelos de classificação para prever o churn de clientes de
cartão de crédito.

## Problema

Classificação binária: prever se um cliente vai cancelar o cartão de crédito
(`Attrition_Flag`: `Existing Customer` vs `Attrited Customer`). O dataset é desbalanceado
(~16% de churn).

## Dataset

Credit Card Customers (BankChurners), carregado diretamente por URL pública, sem necessidade
de upload, login ou token:

```
https://raw.githubusercontent.com/diogorblessa/mvp-credit-card-churn-parte-01/main/data/BankChurners.csv
```

## Como executar

O notebook `notebooks/desenvolvimento_mvp.ipynb` foi pensado para rodar no Google Colab do
início ao fim. Para executar localmente:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/desenvolvimento_mvp.ipynb
```

## Estrutura

```
.
├── notebooks/
│   └── desenvolvimento_mvp.ipynb   # notebook do MVP
├── models/                         # artefato do pipeline final
├── images/                         # gráficos exportados
├── requirements.txt
└── README.md
```

## Status

Este repositório parte de um esqueleto estruturado do notebook (seções, guias e marcações
`# TODO`) que é preenchido manualmente com o código da solução. A versão final, executável de
ponta a ponta, é a entregue no Google Colab.
