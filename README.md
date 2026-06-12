# MVP de Machine Learning & Analytics: Previsão de Churn de Cartão de Crédito (Parte 02)

## 📋 Sobre

Este repositório é o MVP individual da disciplina de Machine Learning & Analytics
(Pós-Graduação em Ciência de Dados e Analytics, PUC-Rio). É a continuação da
[mvp-credit-card-churn-parte-01](https://github.com/diogorblessa/mvp-credit-card-churn-parte-01),
que cobriu a análise exploratória e o pré-processamento. Enquanto a Parte 01 cuidou de entender e
preparar os dados, esta Parte 02 foca em **modelagem, otimização de hiperparâmetros e avaliação**.

O artefato principal é o notebook
[`notebooks/desenvolvimento_mvp.ipynb`](notebooks/desenvolvimento_mvp.ipynb), pensado para rodar
no Google Colab do início ao fim. O problema é uma **classificação binária supervisionada**:
prever se um cliente vai cancelar o cartão de crédito (`Attrition_Flag`: `Existing Customer`
contra `Attrited Customer`). A base descreve `10.127` clientes e `23` atributos, com classes
desbalanceadas (o churn é `16,07%` dos clientes, cerca de `5:1`).

## 🎯 Objetivos de Aprendizado

- Definir um modelo de referência (baseline) e usá-lo como piso a ser superado pelos candidatos
- Treinar e comparar `5` modelos de famílias diferentes (linear, distância e árvore)
- Organizar pré-processamento e treino em um `Pipeline` reprodutível, ajustado só no treino
- Escolher métricas adequadas a uma classe desbalanceada (`F1` e `recall` da classe de churn,
  além do `ROC-AUC`), em vez da acurácia
- Otimizar hiperparâmetros do modelo mais forte com `RandomizedSearchCV` e validação cruzada
- Avaliar o desempenho em um conjunto de teste não visto e discutir erros, limitações e
  prevenção de vazamento de dados
- Salvar o pipeline final como artefato reutilizável

## 📁 Estrutura do Projeto

```text
mvp-credit-card-churn-parte-02/
├── notebooks/
│   └── desenvolvimento_mvp.ipynb     # entrega principal (modelagem, otimização e avaliação)
├── models/
│   ├── comparacao_modelos.csv        # tabela de comparação (validação cruzada)
│   └── metadados_modelo.json         # semente, versões, hiperparâmetros e métricas
├── requirements.txt
└── README.md
```

O pipeline final treinado (`models/modelo_final.pkl`) é gerado pelo próprio notebook na Seção 14
e fica apenas local, porque `*.pkl` está no `.gitignore`. O notebook roda do início ao fim sem
depender de nenhum arquivo pré-existente.

## 🛠️ Tecnologias e Ferramentas

- `Python 3`: linguagem base
- `Jupyter Notebook` / `Google Colab`: ambiente de execução
- `pandas`: manipulação tabular
- `numpy`: operações numéricas
- `matplotlib`: visualizações base
- `seaborn`: gráficos exploratórios
- `scikit-learn`: pipeline, pré-processamento, modelos e validação cruzada
- `xgboost`: modelo de boosting usado como candidato extra
- `joblib`: salvamento e recarga do pipeline final (acompanha o `scikit-learn`)

## 📦 Pré-requisitos

- Ambiente com Python 3
- Acesso à internet durante a execução (os dados são carregados por URL pública)
- Instalação das dependências via `pip install -r requirements.txt` (diferente da Parte 01, aqui
  há um `requirements.txt` no repositório)
- Conhecimentos em pandas, estatística descritiva e fundamentos de classificação supervisionada

## 🚀 Como Acessar a Entrega

### Abrir no Google Colab

Link direto:
<https://colab.research.google.com/github/diogorblessa/mvp-credit-card-churn-parte-02/blob/main/notebooks/desenvolvimento_mvp.ipynb>

### Executar localmente

```bash
pip install -r requirements.txt
jupyter notebook notebooks/desenvolvimento_mvp.ipynb
```

### Execução esperada

1. Abra o notebook (Colab ou Jupyter local)
2. Execute as células sequencialmente, da Seção `1` à Seção `14`
3. Mantenha a ordem para preservar a narrativa e as dependências entre as etapas

> **Observação:** o notebook carrega os dados via URL raw do GitHub, exigindo conexão à internet
> mesmo que exista um arquivo local.

## 📚 Conteúdo Real

| Seção | Conteúdo |
|---|---|
| `1` | Definição do problema, objetivo, tipo de tarefa, premissas e hipóteses |
| `2` | Ambiente, bibliotecas, reprodutibilidade e funções auxiliares |
| `3` | Seleção e carga dos dados, visão geral e dicionário de dados |
| `4` | Análise exploratória mínima (a EDA completa está na Parte 01) |
| `5` | Preparação dos dados e divisão treino/teste |
| `6` | Pré-processamento e pipeline (`ColumnTransformer`) |
| `7` | Baseline e modelos candidatos |
| `8` | Treinamento e avaliação inicial (validação cruzada) |
| `9` | Validação e otimização de hiperparâmetros |
| `10` | Avaliação final no conjunto de teste e análise de erros |
| `11` | Comparação final dos modelos |
| `12` | Boas práticas e rastreabilidade |
| `13` | Conclusão |
| `14` | Salvamento de artefatos e exemplo de uso |

## 🧠 Hipóteses Investigadas

Hipóteses levantadas na Parte 01 e retomadas aqui (Seção `1.4` do notebook):

- **H1**: clientes mais inativos tendem a cancelar mais
- **H2**: clientes que usam pouco o limite do cartão tendem a cancelar mais
- **H3**: clientes com menos produtos contratados no banco tendem a cancelar mais
- **H4**: a faixa de renda influencia a chance de cancelamento
- **H5**: a queda no número de transações ao longo do tempo está ligada ao cancelamento

## 📊 Modelos e Resultados

**Baseline.** O `DummyClassifier` (`most_frequent`) prevê sempre "cliente ativo". Apesar da
acurácia de `83,93%`, fica com `F1 = 0` e `recall = 0` na classe de churn: não identifica nenhum
cancelamento. É o piso que os candidatos precisam superar.

**Comparação dos candidatos (validação cruzada, out-of-fold, 5 folds, só no treino).** Métricas
da classe de churn, vindas de
[`models/comparacao_modelos.csv`](models/comparacao_modelos.csv):

| Modelo | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| `HistGradientBoosting` (otimizado) | `0,9199` | `0,9263` | `0,9136` | `0,9932` |
| `HistGradientBoosting` | `0,9092` | `0,9339` | `0,8857` | `0,9933` |
| `XGBoost` | `0,9081` | `0,9255` | `0,8913` | `0,9926` |
| `RandomForest` | `0,8328` | `0,7558` | `0,9274` | `0,9880` |
| `KNN` | `0,6709` | `0,5676` | `0,8202` | `0,9038` |
| `LogisticRegression` | `0,6423` | `0,8518` | `0,5156` | `0,9259` |
| `Baseline` | `0,0000` | `0,0000` | `0,0000` | `0,5000` |

Os dois modelos de boosting lideraram e quase empataram. O `HistGradientBoosting` foi o melhor
candidato por `F1` out-of-fold (`0,9092`) e seguiu para a otimização.

**Modelo final.** O `HistGradientBoosting` otimizado via `RandomizedSearchCV` (`30` combinações,
validação cruzada de `5` folds, `F1` da classe de churn como métrica-guia). A otimização rendeu
pouco: o `F1` out-of-fold subiu de `0,9092` para `0,9199` (`+0,0107`), ganho da mesma ordem da
variação entre os folds.

**Desempenho no conjunto de teste (dados não vistos).** Métricas da classe de churn, registradas
em [`models/metadados_modelo.json`](models/metadados_modelo.json):

- `F1 = 0,9060`
- `recall = 0,8892` (identifica cerca de `89%` dos cancelamentos)
- `precision = 0,9233` (cerca de `92%` dos sinalizados realmente cancelam)
- `ROC-AUC = 0,9932`

O resultado no teste ficou próximo da estimativa por validação cruzada (`F1 = 0,9199`), com
diferença de apenas `-0,0140`, sinal de que o desempenho se sustenta em dados novos, sem
overfitting relevante. Na matriz de confusão do teste, dos `325` clientes que cancelaram o modelo
identificou `289` e deixou passar `36` (falsos negativos); dos `1.701` clientes ativos, `24`
foram classificados como churn (falsos positivos). O baseline, no mesmo teste, tem acurácia de
`83,96%` sem detectar um único cancelamento.

## 🧪 Decisões de Modelagem e Pré-processamento

- **Remoção de colunas não preditoras (Seção 5).** `CLIENTNUM` (apenas um identificador) e as duas
  colunas de score `Naive_Bayes_Classifier_*` (derivadas do próprio alvo, risco de vazamento)
  são descartadas antes da divisão.
- **Pré-processamento em `ColumnTransformer` (Seção 6).** `StandardScaler` nas `13` colunas
  numéricas e `OneHotEncoder` (`handle_unknown="ignore"`) nas `5` colunas categóricas.
- **Categoria `Unknown` mantida** como categoria informativa, decisão herdada da Parte 01.
- **Sem imputação** (a base tem `0` valores `NaN`) e **sem tratamento de outliers**, porque os
  modelos priorizados são de árvore (insensíveis a escala e a valores extremos) e os outliers
  costumam ser comportamento real associado ao churn.
- **Divisão treino/teste `80/20` estratificada** pelo alvo (`stratify=y`), mantendo os mesmos
  `~16%` de churn nos dois conjuntos. O pipeline é ajustado apenas com o treino, e o teste fica
  reservado até a avaliação final (Seção 10), evitando vazamento.
- **Desbalanceamento** compensado com `class_weight="balanced"` (`LogisticRegression`,
  `RandomForest`, `HistGradientBoosting`) e `scale_pos_weight` (`XGBoost`).
- **Otimização (Seção 9).** `RandomizedSearchCV` ajustou `5` hiperparâmetros do
  `HistGradientBoosting` (`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`,
  `l2_regularization`), guiado pelo `F1` da classe de churn em validação cruzada de `5` folds.
- **Reprodutibilidade.** Uma única semente é fixada na Seção 2 e reutilizada na divisão, na
  validação cruzada e em todos os modelos; as versões das bibliotecas ficam registradas.

## 🔗 Conexões com a Formação

Esta entrega fecha o ciclo iniciado na Parte 01: a preparação dos dados (análise exploratória,
codificação, tratamento de `Unknown` e divisão) serve de base para a etapa de modelagem
supervisionada, otimização e avaliação desta Parte 02, conectando manipulação de dados com a
construção e a validação de um classificador de churn.

## 📖 Recursos Adicionais

- Parte 01 (EDA e pré-processamento):
  <https://github.com/diogorblessa/mvp-credit-card-churn-parte-01>
- Notebook principal: [`notebooks/desenvolvimento_mvp.ipynb`](notebooks/desenvolvimento_mvp.ipynb)
- Dataset (URL raw pública):
  <https://raw.githubusercontent.com/diogorblessa/mvp-credit-card-churn-parte-01/main/data/BankChurners.csv>
- Kaggle: <https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers>
- scikit-learn: <https://scikit-learn.org/stable/>
- XGBoost: <https://xgboost.readthedocs.io/>
- pandas: <https://pandas.pydata.org/docs/>
