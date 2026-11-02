# SHAP 2017 paper: final experiment code package

This package contains runnable, educational reproductions inspired by:

**Scott M. Lundberg and Su-In Lee (2017), “A Unified Approach to Interpreting Model Predictions.”**
https://arxiv.org/abs/1705.07874

It combines transparent toy-game implementations with selected demonstrations using the official `shap` Python library. It is **not the authors' original experiment code** and does not claim exact numerical reproduction of every paper figure or human-subject result.

## 1. Installation

Python 3.9+ is recommended. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

For the MNIST DeepExplainer experiment, also install TensorFlow:

```bash
pip install -r requirements-mnist.txt
```

If TensorFlow installation is not practical, simply skip the MNIST experiment.

## 2. Run experiments

Run one experiment at a time:

```bash
python shap_paper_experiments.py --only fever
python shap_paper_experiments.py --only dependence
python shap_paper_experiments.py --only linear
python shap_paper_experiments.py --only tree
python shap_paper_experiments.py --only human
python shap_paper_experiments.py --only mnist
```

Run all experiments (MNIST requires TensorFlow and trains a small CNN):

```bash
python shap_paper_experiments.py --only all
```

The scripts save plots in the current working directory and display them.

## 3. Experiment guide

| Experiment | What it demonstrates | Status |
|---|---|---|
| `fever` | Exact Shapley values for the paper's fever/cough toy function; marginal effects depend on the other symptom | Transparent toy-game reproduction |
| `dependence` | Synthetic correlated binary features; compares conditional and interventional coalition-value definitions using exact Shapley enumeration | Synthetic demonstration, not a paper dataset |
| `linear` | Official SHAP `LinearExplainer` with an independent masker versus a correlation-aware `Impute` masker | Uses the SHAP library; API may vary by SHAP version |
| `tree` | Kernel SHAP, Shapley Sampling and LIME on a synthetic decision-tree classifier | Modern illustrative comparison, not exact original data/settings |
| `human` | Theoretical Shapley allocations for toy tasks; optional entry of actual classroom votes | Does not fabricate the paper's participant data |
| `mnist` | Trains a compact CNN and explains four MNIST images with `shap.DeepExplainer` | Follows the official SHAP repository MNIST workflow pattern; fresh model/data run, not the original checkpoint |

## 4. What to say in your presentation

Be precise about the phrase “features are not independent”:

- **Statistical feature dependence** means the input variables are correlated in the data distribution. The `dependence` and `linear` experiments address this.
- **Model interaction** means the effect of one feature on the model output changes depending on another feature's value. The fever/cough toy function demonstrates this.
- **Attribution dependence** means the attribution assigned to a feature can change when the coalition-value definition or assumptions about missing features change.

These are related but not interchangeable. The fever/cough example demonstrates interaction; it does not, by itself, establish statistical correlation between fever and cough.

## 5. Relationship to the official SHAP repository

Official repository: https://github.com/shap/shap

The `linear` and `mnist` experiments use the maintained SHAP library. The MNIST experiment follows the repository's DeepExplainer notebook pattern: train/load a Keras image classifier, select a background batch, construct `shap.DeepExplainer`, calculate SHAP values and visualize them with `shap.image_plot`.

The repository has evolved since 2017. Some current explainers and capabilities (for example, TreeExplainer interaction-value functionality) should not be presented as experiments that appeared in the original 2017 paper.

## 6. Notes and limitations

- The original paper's reported human-study results require real participants; this package does not invent participant responses.
- The tree comparison is a reproducible teaching example, not a reconstruction of the paper's exact repeated trials, datasets, random seeds or percentile plots.
- The MNIST CNN is trained locally and may produce different explanations across runs, software versions and hardware.
- SHAP APIs evolve. If a method signature changes in a future release, check https://shap.readthedocs.io/ and the official repository.
