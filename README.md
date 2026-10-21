# Classificação de Séries Temporais para Detecção de Falhas

Este projeto reúne um pipeline completo para **detectar falhas em equipamentos a partir de dados de sensores**. A ideia é olhar para trechos curtos das séries temporais (por exemplo, a última hora de pressão, vazão, temperatura, vibração...) e decidir se aquele trecho corresponde a uma **operação normal** ou a um **período de falha**.

Em vez de apostar em um único algoritmo, o repositório implementa e compara três famílias de modelos, todas usando os mesmos dados de treino e teste:

| Abordagem | Modelos | Ideia |
|---|---|---|
| Features estatísticas + ML clássico | Random Forest, XGBoost, kNN | Resume cada janela em estatísticas (média, desvio, quantis, autocorrelação...) e usa classificadores tabulares. Fácil de interpretar com SHAP. |
| Kernels aleatórios | ROCKET, MiniROCKET | Aplica milhares de convoluções aleatórias e treina um classificador linear sobre o resultado. Rápido e muito competitivo. |
| Deep learning (PyTorch) | CNN 1D, InceptionTime, ResNet 1D, ResNet 2D | Redes neurais que aprendem as features diretamente da janela. |

Tudo é demonstrado em **um único notebook template**, que roda imediatamente com um **dataset sintético** e pode ser adaptado aos seus dados mudando apenas uma célula de configuração.

## Como funciona

```
 CSV de sensores ──► remove períodos com     ──► rotula períodos ──► divide treino/teste
 (1 coluna por        o equipamento desligado      de falha            pela data
  variável)           e transitórios

                                ┌─► features estatísticas ─► RF / XGBoost / kNN
 janelas deslizantes ───────────┼─► ROCKET / MiniROCKET ───► RidgeClassifierCV      ──► comparação
 (ex.: 60 min, passo 30 min)    └─► janela normalizada ────► CNN / InceptionTime /      no teste
                                                             ResNet (PyTorch)
```

1. **Pré-processamento.** Períodos em que o equipamento está desligado (por exemplo, corrente do motor abaixo de um limite), junto com alguns minutos antes e depois (partida e parada), são removidos: eles não representam nem operação normal nem falha.
2. **Rotulagem.** Você informa uma lista de períodos `(início, fim)` em que houve falha; cada amostra dentro desses períodos recebe o rótulo `1`.
3. **Divisão temporal.** O treino usa o passado e o teste usa o futuro, como aconteceria em produção. Uma divisão aleatória colocaria janelas quase idênticas nos dois conjuntos e mostraria um desempenho irrealmente alto.
4. **Janelamento.** A série é cortada em janelas sobrepostas. Cada janela é um exemplo de classificação e é rotulada como falha quando mais da metade das suas amostras é falha. Janelas que atravessam lacunas no tempo (trechos removidos) são descartadas.
5. **Modelagem e avaliação.** As três abordagens são treinadas e avaliadas nas mesmas janelas de teste.

### Sobre as métricas

Falhas costumam ser raras: em um teste com 1% de janelas de falha, um modelo que **nunca** detecta falha tem 99% de acurácia. Por isso o projeto reporta métricas da classe de falha (**precision**, **recall** e **F1**), a **acurácia balanceada** e métricas de ranqueamento (**ROC AUC** e **PR AUC**). A PR AUC é a mais informativa quando as classes são muito desbalanceadas.

## Estrutura do repositório

```
.
├── notebooks/
│   └── tsc_template.ipynb   # notebook de exemplo: o pipeline completo, do CSV à comparação
├── tsc/                     # código reutilizável importado pelo notebook
│   ├── data.py              # leitura do CSV, remoção de períodos desligado, rotulagem, dados sintéticos
│   ├── windowing.py         # janelas deslizantes, divisão treino/validação, normalização
│   ├── features.py          # features estatísticas por janela
│   ├── models.py            # arquiteturas PyTorch (CNN1D, InceptionTime, ResNet1D, ResNet2D)
│   ├── training.py          # loop de treino (focal loss, AdamW, checkpoint do melhor modelo)
│   └── evaluation.py        # métricas e gráficos
├── pyproject.toml           # dependências
└── uv.lock
```

## Instalação

O projeto usa Python 3.12 e o gerenciador [uv](https://docs.astral.sh/uv/):

```bash
git clone <url-deste-repositorio>
cd timeseries-classification
uv sync
```

O `uv sync` cria o ambiente virtual em `.venv/`, instala as dependências (PyTorch, scikit-learn, sktime, XGBoost, SHAP...) e instala o pacote `tsc` em modo editável.

Sem o uv, também é possível usar pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . ipykernel nbformat
```

O treino das redes usa automaticamente GPU NVIDIA (CUDA) ou Apple Silicon (MPS) quando disponíveis, e CPU caso contrário.

## Como usar

### 1. Rodar o exemplo com dados sintéticos

Abra `notebooks/tsc_template.ipynb` no Jupyter ou no VS Code, selecione o kernel do `.venv` e execute todas as células. Com `USE_SYNTHETIC_DATA = True` (o padrão), o notebook gera 90 dias de dados de seis sensores com paradas do equipamento e falhas simuladas, e roda o pipeline inteiro em poucos minutos.

### 2. Usar com os seus dados

Coloque seu arquivo em `data/` (essa pasta é ignorada pelo git) no formato abaixo: separador `;`, uma coluna de timestamp e uma coluna numérica por variável.

```
Timestamp;corrente_motor;pressao;vazao;temperatura
2024-01-01 00:00:00;51.2;3.04;121.7;70.1
2024-01-01 00:01:00;50.8;3.06;120.9;70.2
...
```

Depois, edite apenas a célula **Configuração** do notebook:

| Parâmetro | O que define |
|---|---|
| `USE_SYNTHETIC_DATA`, `DATA_PATH` | Usar dados sintéticos ou ler o seu CSV |
| `TIMESTAMP_COL`, `DROP_COLUMNS`, `SAMPLING_FREQ` | Nome da coluna de tempo, colunas a descartar e período de amostragem |
| `FAULT_PERIODS` | Lista de `(início, fim)` dos períodos de falha conhecidos |
| `OFF_FILTER` | Variável e limite que indicam equipamento desligado, e quantos minutos cortar antes/depois (`None` desativa) |
| `TRAIN_PERIOD`, `TEST_PERIOD` | Intervalos de datas de treino e de teste |
| `WINDOW_SIZE`, `STEP` | Tamanho das janelas e o passo entre elas (em amostras) |
| `DL_MODELS`, `EPOCHS`, `BATCH_SIZE`, `VAL_FRACTION` | Quais redes treinar e como |

Os pesos do melhor modelo de cada rede são salvos em `artifacts/`.

### 3. Usar o pacote fora do notebook

As funções do pacote `tsc` também podem ser usadas em scripts:

```python
from tsc.data import make_synthetic_dataset, remove_off_periods, label_periods, select_period
from tsc.windowing import make_windows, temporal_split, fit_window_scaler, scale_windows
from tsc.models import ResNet1D
from tsc.training import train_model, predict_proba

df, fault_periods = make_synthetic_dataset()
df = remove_off_periods(df, "corrente_motor", threshold=5, pre_cut=30, post_cut=30)
df = label_periods(df, fault_periods)
features = [c for c in df.columns if c not in ("Timestamp", "Falhas")]

X, y, _ = make_windows(select_period(df, "2024-01-01", "2024-03-05 23:59"), features)
X_fit, X_val, y_fit, y_val = temporal_split(X, y, val_fraction=0.2, gap=2)
scaler = fit_window_scaler(X_fit)

result = train_model(
    ResNet1D(n_features=len(features)),
    scale_windows(scaler, X_fit), y_fit,
    scale_windows(scaler, X_val), y_val,
    epochs=20,
)
probabilities = predict_proba(result.best_model, scale_windows(scaler, X_val))
```

## Tecnologias

- [PyTorch](https://pytorch.org/): redes neurais
- [scikit-learn](https://scikit-learn.org/) e [XGBoost](https://xgboost.readthedocs.io/): modelos clássicos e métricas
- [sktime](https://www.sktime.net/): ROCKET e MiniROCKET
- [SHAP](https://shap.readthedocs.io/): interpretação dos modelos baseados em árvores
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) e [SciPy](https://scipy.org/): manipulação de dados e estatísticas
- [Plotly](https://plotly.com/python/) e [Matplotlib](https://matplotlib.org/): visualização

## Referências

- Dempster, Petitjean e Webb. *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels* (2020).
- Dempster, Schmidt e Webb. *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification* (2021).
- Ismail Fawaz et al. *InceptionTime: Finding AlexNet for Time Series Classification* (2020).
- Wang, Yan e Oates. *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline* (2017).
- Lin et al. *Focal Loss for Dense Object Detection* (2017).
