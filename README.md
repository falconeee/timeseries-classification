# Time Series Classification for Fault Detection

This project provides a complete pipeline to **detect equipment faults from sensor data**. The idea is to look at short stretches of the time series (for example, the last hour of pressure, flow, temperature, vibration...) and decide whether that stretch corresponds to **normal operation** or to a **fault period**.

Instead of betting on a single algorithm, the repository implements and compares three model families, all using the same training and test data:

| Approach | Models | Idea |
|---|---|---|
| Statistical features + classic ML | Random Forest, XGBoost, kNN | Summarizes each window into statistics (mean, standard deviation, quantiles, autocorrelation...) and uses tabular classifiers. Easy to interpret with SHAP. |
| Random kernels | ROCKET, MiniROCKET | Applies thousands of random convolutions and trains a linear classifier on the result. Fast and very competitive. |
| Deep learning (PyTorch) | 1D CNN, InceptionTime, 1D ResNet, 2D ResNet | Neural networks that learn the features directly from the window. |

Everything is demonstrated in **a single template notebook**, which runs out of the box on a **synthetic dataset** and can be adapted to your data by changing just one configuration cell.

## How it works

```
 Sensor CSV       ──► remove periods with      ──► label fault ──► split train/test
 (1 column per         the equipment off and        periods          by date
  variable)            transients

                                ┌─► statistical features ─► RF / XGBoost / kNN
 sliding windows ───────────────┼─► ROCKET / MiniROCKET ──► RidgeClassifierCV      ──► comparison
 (e.g. 60 min, 30 min step)     └─► normalized window ────► CNN / InceptionTime /      on the test set
                                                            ResNet (PyTorch)
```

1. **Preprocessing.** Periods when the equipment is off (for example, motor current below a threshold), together with a few minutes before and after (start-up and shutdown), are removed: they represent neither normal operation nor a fault.
2. **Labeling.** You provide a list of `(start, end)` periods when a fault happened; every sample inside those periods gets label `1`.
3. **Temporal split.** Training uses the past and testing uses the future, as would happen in production. A random split would put nearly identical windows in both sets and show unrealistically high performance.
4. **Windowing.** The series is cut into overlapping windows. Each window is one classification example and is labeled as a fault when more than half of its samples are faults. Windows that cross time gaps (removed stretches) are discarded.
5. **Modeling and evaluation.** The three approaches are trained and evaluated on the same test windows.

### About the metrics

Faults are usually rare: on a test set with 1% fault windows, a model that **never** detects a fault has 99% accuracy. That is why the project reports metrics for the fault class (**precision**, **recall** and **F1**), **balanced accuracy** and ranking metrics (**ROC AUC** and **PR AUC**). PR AUC is the most informative when the classes are highly imbalanced.

## Repository structure

```
.
├── notebooks/
│   └── tsc_template.ipynb   # example notebook: the full pipeline, from CSV to comparison
├── tsc/                     # reusable code imported by the notebook
│   ├── data.py              # CSV loading, removal of off periods, labeling, synthetic data
│   ├── windowing.py         # sliding windows, train/validation split, normalization
│   ├── features.py          # per-window statistical features
│   ├── models.py            # PyTorch architectures (CNN1D, InceptionTime, ResNet1D, ResNet2D)
│   ├── training.py          # training loop (focal loss, AdamW, best-model checkpoint)
│   └── evaluation.py        # metrics and plots
├── pyproject.toml           # dependencies
└── uv.lock
```

## Installation

The project uses Python 3.12 and the [uv](https://docs.astral.sh/uv/) package manager:

```bash
git clone <this-repository-url>
cd timeseries-classification
uv sync
```

`uv sync` creates the virtual environment in `.venv/`, installs the dependencies (PyTorch, scikit-learn, sktime, XGBoost, SHAP...) and installs the `tsc` package in editable mode.

Without uv, you can also use pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . ipykernel nbformat
```

Network training automatically uses an NVIDIA GPU (CUDA) or Apple Silicon (MPS) when available, and the CPU otherwise.

## How to use

### 1. Run the example with synthetic data

Open `notebooks/tsc_template.ipynb` in Jupyter or VS Code, select the `.venv` kernel and run all cells. With `USE_SYNTHETIC_DATA = True` (the default), the notebook generates 90 days of data from six sensors with equipment stops and simulated faults, and runs the whole pipeline in a few minutes.

### 2. Use it with your own data

Put your file in `data/` (this folder is ignored by git) in the format below: `;` separator, one timestamp column and one numeric column per variable.

```
Timestamp;corrente_motor;pressao;vazao;temperatura
2024-01-01 00:00:00;51.2;3.04;121.7;70.1
2024-01-01 00:01:00;50.8;3.06;120.9;70.2
...
```

Then edit only the notebook's **Configuração** (configuration) cell:

| Parameter | What it sets |
|---|---|
| `USE_SYNTHETIC_DATA`, `DATA_PATH` | Use synthetic data or read your CSV |
| `TIMESTAMP_COL`, `DROP_COLUMNS`, `SAMPLING_FREQ` | Name of the time column, columns to drop and sampling period |
| `FAULT_PERIODS` | List of `(start, end)` of the known fault periods |
| `OFF_FILTER` | Variable and threshold that indicate the equipment is off, and how many minutes to cut before/after (`None` disables it) |
| `TRAIN_PERIOD`, `TEST_PERIOD` | Training and test date ranges |
| `WINDOW_SIZE`, `STEP` | Window size and the step between windows (in samples) |
| `DL_MODELS`, `EPOCHS`, `BATCH_SIZE`, `VAL_FRACTION` | Which networks to train and how |

The weights of each network's best model are saved in `artifacts/`.

### 3. Use the package outside the notebook

The functions of the `tsc` package can also be used in scripts:

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

## Technologies

- [PyTorch](https://pytorch.org/): neural networks
- [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/): classic models and metrics
- [sktime](https://www.sktime.net/): ROCKET and MiniROCKET
- [SHAP](https://shap.readthedocs.io/): interpretation of the tree-based models
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/): data handling and statistics
- [Plotly](https://plotly.com/python/) and [Matplotlib](https://matplotlib.org/): visualization

## References

- Dempster, Petitjean and Webb. *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels* (2020).
- Dempster, Schmidt and Webb. *MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification* (2021).
- Ismail Fawaz et al. *InceptionTime: Finding AlexNet for Time Series Classification* (2020).
- Wang, Yan and Oates. *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline* (2017).
- Lin et al. *Focal Loss for Dense Object Detection* (2017).
