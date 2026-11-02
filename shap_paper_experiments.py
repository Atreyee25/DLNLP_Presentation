"""
Educational reproductions inspired by:
Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
https://arxiv.org/abs/1705.07874

This is NOT the authors' original code or an exact reproduction of every figure.
It recreates the core experimental ideas with transparent, runnable Python.

Experiments:
1. Fever/cough interaction example (paper Figure 4A).
2. Conditional vs interventional feature handling for correlated features.
3. Tree-model estimator comparison: Kernel SHAP, Shapley sampling, LIME.
4. Three-person max-profit allocation (paper Figure 4B), plus optional human voting.
5. MNIST DeepExplainer demonstration following the official SHAP repository notebook pattern.
6. Correlation-aware LinearExplainer comparison using SHAP maskers.

Install:
    pip install -r requirements.txt

Run:
    python shap_paper_experiments.py
    python shap_paper_experiments.py --only dependence
    python shap_paper_experiments.py --only tree
    python shap_paper_experiments.py --only mnist
    python shap_paper_experiments.py --only linear
"""

import argparse
import itertools
import math
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Shared utilities: exact Shapley values for a small number of features
# ---------------------------------------------------------------------

def powerset_without(feature, n_features):
    others = [j for j in range(n_features) if j != feature]
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            yield tuple(subset)


def shapley_from_value_function(value_fn, n_features):
    """
    Exact Shapley values from a coalition-value function v(S).
    v_fn accepts a tuple of feature indices and returns a scalar.
    """
    phi = np.zeros(n_features, dtype=float)
    factorial = math.factorial
    denom = factorial(n_features)

    for i in range(n_features):
        for S in powerset_without(i, n_features):
            k = len(S)
            weight = factorial(k) * factorial(n_features - k - 1) / denom
            marginal = value_fn(tuple(sorted(S + (i,)))) - value_fn(S)
            phi[i] += weight * marginal

    return phi


def print_attributions(names, baseline, prediction, phi):
    print("\nAttributions")
    print("-" * 52)
    print(f"{'Feature':<18}{'SHAP value':>14}")
    for name, value in zip(names, phi):
        print(f"{name:<18}{value:>14.4f}")
    print("-" * 52)
    print(f"{'Baseline':<18}{baseline:>14.4f}")
    print(f"{'Sum / prediction':<18}{baseline + phi.sum():>14.4f}")
    print(f"{'Model prediction':<18}{prediction:>14.4f}")


# ---------------------------------------------------------------------
# Experiment 1: fever/cough interaction (paper Figure 4A)
# ---------------------------------------------------------------------

def sickness_score(fever, cough):
    """Paper's toy model: 2 if both symptoms; 5 if exactly one; 0 if neither."""
    if fever and cough:
        return 2.0
    if fever or cough:
        return 5.0
    return 0.0


def experiment_fever_cough():
    print("\nEXPERIMENT 1 — Fever/cough interaction")
    # Explain the instance fever=1, cough=1.
    x = (1, 1)
    names = ["Fever", "Cough"]

    # Paper's toy study uses a baseline of 0 (neither symptom).
    def v(S):
        fever = x[0] if 0 in S else 0
        cough = x[1] if 1 in S else 0
        return sickness_score(fever, cough)

    baseline = v(())
    prediction = sickness_score(*x)
    phi = shapley_from_value_function(v, 2)
    print_attributions(names, baseline, prediction, phi)

    # Visualize interaction: marginal effect depends on whether other symptom is present.
    fever_without_cough = sickness_score(1, 0) - sickness_score(0, 0)
    fever_with_cough = sickness_score(1, 1) - sickness_score(0, 1)
    cough_without_fever = sickness_score(0, 1) - sickness_score(0, 0)
    cough_with_fever = sickness_score(1, 1) - sickness_score(1, 0)

    print("\nMarginal effect of Fever:")
    print(f"  when Cough absent:  {fever_without_cough:+.1f}")
    print(f"  when Cough present: {fever_with_cough:+.1f}")
    print("Marginal effect of Cough:")
    print(f"  when Fever absent:  {cough_without_fever:+.1f}")
    print(f"  when Fever present: {cough_with_fever:+.1f}")

    labels = ["Fever effect\n(cough=0)", "Fever effect\n(cough=1)",
              "Cough effect\n(fever=0)", "Cough effect\n(fever=1)"]
    vals = [fever_without_cough, fever_with_cough,
            cough_without_fever, cough_with_fever]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, vals)
    plt.axhline(0, linewidth=0.8)
    plt.ylabel("Change in sickness score")
    plt.title("Interaction: a feature's marginal contribution depends on the other feature")
    plt.tight_layout()
    plt.savefig("exp1_fever_cough_interaction.png", dpi=160)
    plt.show()


# ---------------------------------------------------------------------
# Experiment 2: correlated features and conditional/interventional values
# ---------------------------------------------------------------------

def make_correlated_binary_data(n=5000, seed=7):
    """
    Two positively correlated binary features.
    X1 ~ Bernoulli(.5); X2 copies X1 with probability .9.
    A third feature is independent noise.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.binomial(1, 0.5, n)
    flip = rng.binomial(1, 0.1, n)
    x2 = np.logical_xor(x1, flip).astype(int)
    x3 = rng.binomial(1, 0.5, n)
    return np.column_stack([x1, x2, x3])


def black_box_model(X):
    """
    Nonlinear model with both an interaction and a correlated-feature effect.
    X columns: x1, x2, x3.
    """
    X = np.asarray(X)
    return 2.0 * X[:, 0] + 2.0 * X[:, 1] - 3.0 * X[:, 0] * X[:, 1] + 1.5 * X[:, 2]


def experiment_dependence():
    print("\nEXPERIMENT 2 — Conditional vs interventional feature handling")
    background = make_correlated_binary_data()
    # Explain x1=1, x2=1, x3=0. Use an actually observed row with this pattern.
    candidates = np.where((background[:, 0] == 1) &
                         (background[:, 1] == 1) &
                         (background[:, 2] == 0))[0]
    if len(candidates) == 0:
        raise RuntimeError("Could not find requested example; increase sample size.")
    x = background[candidates[0]].copy()
    names = ["X1", "X2", "X3"]
    n_features = background.shape[1]

    # Interventional game: hold present features fixed and sample missing ones
    # from their marginal background distribution.
    def v_interventional(S):
        hybrid = background.copy()
        for j in S:
            hybrid[:, j] = x[j]
        return float(np.mean(black_box_model(hybrid)))

    # Conditional game: average predictions over background rows that match
    # the observed values of features in S. This empirical conditional estimate
    # is exact for the discrete synthetic data up to sampling error.
    def v_conditional(S):
        if len(S) == 0:
            rows = background
        else:
            mask = np.ones(len(background), dtype=bool)
            for j in S:
                mask &= (background[:, j] == x[j])
            rows = background[mask]
            if len(rows) == 0:
                # Not expected for this generated discrete example.
                return v_interventional(S)
        return float(np.mean(black_box_model(rows)))

    phi_int = shapley_from_value_function(v_interventional, n_features)
    phi_cond = shapley_from_value_function(v_conditional, n_features)
    base_int = v_interventional(())
    base_cond = v_conditional(())
    pred = float(black_box_model(x.reshape(1, -1))[0])

    print(f"Explained input: {dict(zip(names, x.tolist()))}")
    print(f"Model prediction: {pred:.4f}")
    print("\nInterventional (marginal replacement) game:")
    print_attributions(names, base_int, pred, phi_int)
    print("\nConditional (dependence-aware) game:")
    print_attributions(names, base_cond, pred, phi_cond)

    pos = np.arange(n_features)
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(pos - width/2, phi_int, width, label="Interventional")
    plt.bar(pos + width/2, phi_cond, width, label="Conditional")
    plt.xticks(pos, names)
    plt.ylabel("Attribution")
    plt.title("Correlated features: attribution depends on the missing-feature definition")
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp2_conditional_vs_interventional.png", dpi=160)
    plt.show()

    corr = np.corrcoef(background[:, 0], background[:, 1])[0, 1]
    print(f"\nEmpirical correlation corr(X1, X2) = {corr:.3f}")
    print("Interpretation: different coalition-value definitions can yield different attributions.")
    print("This is distinct from the fever/cough example, which demonstrates model interaction.")


# ---------------------------------------------------------------------
# Experiment 3: tree model — Kernel SHAP vs sampling SHAP vs LIME
# Requires: shap, lime, scikit-learn
# ---------------------------------------------------------------------

def experiment_tree_comparison():
    print("\nEXPERIMENT 3 — Tree-model explanation method comparison")
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.datasets import make_classification
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        print("Install optional packages with: pip install shap lime scikit-learn")
        raise exc

    X, y = make_classification(
        n_samples=1200, n_features=10, n_informative=5, n_redundant=2,
        n_repeated=0, n_classes=2, random_state=12, shuffle=False
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=12)
    model.fit(X, y)
    x = X[0:1]
    background = shap.sample(X, 100, random_state=1)

    # Kernel SHAP (model-agnostic weighted-regression estimator).
    kernel = shap.KernelExplainer(model.predict_proba, background)
    kernel_values = kernel.shap_values(x, nsamples=300)

    # Shapley sampling approximation.
    sampler = shap.SamplingExplainer(model.predict_proba, background)
    sampling_values = sampler.shap_values(x, nsamples=300)

    # LIME local surrogate.
    lime_exp = LimeTabularExplainer(
        X, feature_names=[f"Feature {i}" for i in range(X.shape[1])],
        class_names=["class 0", "class 1"], mode="classification",
        discretize_continuous=False, random_state=1
    )
    lime_result = lime_exp.explain_instance(
        x[0], model.predict_proba, num_features=X.shape[1], num_samples=3000
    )

    # Handle SHAP versions that return list-per-class or ndarray.
    def class_one_values(values):
        if isinstance(values, list):
            return np.asarray(values[1]).reshape(-1)
        arr = np.asarray(values)
        if arr.ndim == 3:
            return arr[0, :, 1]
        if arr.ndim == 2:
            return arr.reshape(-1)
        return arr.reshape(-1)

    k = class_one_values(kernel_values)
    ss = class_one_values(sampling_values)
    lime_map = dict(lime_result.as_list(label=1))
    lime_vals = np.array([lime_map.get(f"Feature {i}", 0.0) for i in range(X.shape[1])])

    # LIME labels often include intervals when discretization is on; here it is off.
    names = [f"F{i}" for i in range(X.shape[1])]
    idx = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(idx - width, k, width, label="Kernel SHAP")
    plt.bar(idx, ss, width, label="Shapley Sampling")
    plt.bar(idx + width, lime_vals, width, label="LIME")
    plt.xticks(idx, names)
    plt.ylabel("Local contribution (class 1)")
    plt.title("One-instance explanation comparison on a decision tree")
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp3_kernel_sampling_lime.png", dpi=160)
    plt.show()

    print("Model probability:", model.predict_proba(x)[0])
    print("Note: this is a modern illustrative recreation, not the exact paper's")
    print("original tree/data/random-seed setup or its 200-replicate percentile plot.")


# ---------------------------------------------------------------------
# Experiment 4: human allocation toy tasks (paper Figure 4A and 4B)
# ---------------------------------------------------------------------

def exact_game_values_from_truth_table(players, payoff_fn):
    def v(S):
        state = [0] * players
        for i in S:
            state[i] = 1
        return float(payoff_fn(state))
    return shapley_from_value_function(v, players), v(()), v(tuple(range(players)))


def experiment_human_allocation(collect_votes=False, n_participants=30):
    print("\nEXPERIMENT 4 — Human-credit allocation toy tasks")
    # Task A: Fever/cough output is 2 when both symptoms are present.
    phi_a, base_a, full_a = exact_game_values_from_truth_table(
        2, lambda z: sickness_score(z[0], z[1])
    )
    print(f"Task A (fever/cough): baseline={base_a:.1f}, output={full_a:.1f}, SHAP={phi_a}")

    # Task B: profit = max(5,4,0) = 5, with each player's score as the input.
    # The cooperative game value is max of the scores in the coalition; empty coalition = 0.
    scores = [5.0, 4.0, 0.0]
    def max_game(S):
        return max([scores[i] for i in S], default=0.0)
    phi_b = shapley_from_value_function(max_game, 3)
    print(f"Task B (max-profit): scores={scores}, SHAP allocation={phi_b.tolist()}")

    if collect_votes:
        print("\nOptional small classroom replication.")
        print("Ask each participant to allocate exactly 2 points for Task A (output=2).")
        print("Enter two numbers separated by spaces, e.g. '1 1'.")
        votes = []
        for p in range(n_participants):
            raw = input(f"Participant {p+1}: ").strip().split()
            if len(raw) != 2:
                print("Invalid entry; skipping.")
                continue
            try:
                v = np.array([float(raw[0]), float(raw[1])])
                if np.any(v < 0) or not np.isclose(v.sum(), 2.0):
                    print("Must be nonnegative and sum to 2; skipping.")
                    continue
                votes.append(tuple(v))
            except ValueError:
                print("Invalid numbers; skipping.")
        if votes:
            print("Most common human allocation:", Counter(votes).most_common(1)[0])
            print("SHAP allocation:", tuple(np.round(phi_a, 3)))
    print("Important: the paper's reported participant responses cannot be recreated")
    print("without recruiting participants; this script computes theoretical allocations.")


# ---------------------------------------------------------------------
# Experiment 5: MNIST DeepExplainer (official SHAP repository pattern)
# Requires: tensorflow and shap
# ---------------------------------------------------------------------

def experiment_mnist():
    print("\nEXPERIMENT 5 — MNIST with SHAP DeepExplainer")
    try:
        import shap
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        print("Install optional packages with: pip install tensorflow shap")
        raise exc

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Compact CNN trained here for a runnable demonstration. The SHAP workflow
    # mirrors the official repository's MNIST DeepExplainer notebook, but the
    # network/checkpoint and selected images are not claimed to be identical.
    model = keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train[:20000, ..., None], y_train[:20000],
              epochs=2, batch_size=128, validation_split=0.1, verbose=1)

    # Use a small background set, as in the official DeepExplainer example.
    rng = np.random.default_rng(7)
    background_indices = rng.choice(len(x_train), size=100, replace=False)
    background = x_train[background_indices, ..., None]
    test_images = x_test[1:5, ..., None]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_images)

    # SHAP versions differ: older releases return a list per class; newer
    # releases may return one ndarray with the output dimension at the end.
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5 and shap_values.shape[-1] == 10:
            shap_values_for_plot = [shap_values[..., i] for i in range(10)]
        else:
            shap_values_for_plot = shap_values
    else:
        shap_values_for_plot = shap_values

    predictions = np.argmax(model.predict(test_images, verbose=0), axis=1)
    print("Predicted classes for the four explained images:", predictions.tolist())
    print("Expected values:", np.asarray(explainer.expected_value).reshape(-1).tolist())
    shap.image_plot(shap_values_for_plot, -test_images)
    print("This follows the official SHAP MNIST DeepExplainer workflow pattern.")
    print("It is a fresh run with a newly trained compact CNN, not a bit-for-bit")
    print("reproduction of the paper's original checkpoint or exact Figure 5 settings.")


# ---------------------------------------------------------------------
# Experiment 6: LinearExplainer with independent vs correlation-aware masker
# Requires: shap and scikit-learn
# ---------------------------------------------------------------------

def experiment_linear():
    print("\nEXPERIMENT 6 — SHAP LinearExplainer: independent vs correlated features")
    try:
        import shap
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        print("Install with: pip install shap scikit-learn")
        raise exc

    X = make_correlated_binary_data(n=4000, seed=21).astype(float)
    # A linear response keeps this example appropriate for LinearExplainer.
    y = ((2.0 * X[:, 0] + 1.8 * X[:, 1] + 0.8 * X[:, 2]) > 2.0).astype(int)
    model = LogisticRegression().fit(X, y)
    x = X[[0]]

    # Independent masker ignores feature correlation when replacing missing data.
    independent_masker = shap.maskers.Independent(X)
    independent_explainer = shap.LinearExplainer(model, independent_masker)
    independent_values = independent_explainer(x)

    # Impute masker uses the background covariance structure to account for
    # dependencies among features in the linear explanation.
    impute_masker = shap.maskers.Impute(X)
    correlation_explainer = shap.LinearExplainer(model, impute_masker)
    correlation_values = correlation_explainer(x)

    def one_output_values(explanation):
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            return values[0, :, 1]
        if values.ndim == 2:
            return values[0]
        return values.reshape(-1)

    phi_ind = one_output_values(independent_values)
    phi_corr = one_output_values(correlation_values)
    names = ["X1", "X2", "X3"]
    print("Empirical corr(X1, X2):", float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))
    print("Input:", x[0].tolist())
    print("Independent SHAP values:", phi_ind.tolist())
    print("Correlation-aware SHAP values:", phi_corr.tolist())

    pos = np.arange(len(names)); width = 0.36
    plt.figure(figsize=(8, 4.5))
    plt.bar(pos - width/2, phi_ind, width, label="Independent features")
    plt.bar(pos + width/2, phi_corr, width, label="Correlation-aware (Impute)")
    plt.xticks(pos, names)
    plt.ylabel("SHAP value (class 1 / model output)")
    plt.title("LinearExplainer: feature dependence changes attribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp6_linear_independent_vs_correlation_aware.png", dpi=160)
    plt.show()

    print("Interpretation: the two explainers use different assumptions about")
    print("missing features. This is an illustration of attribution dependence,")
    print("not proof that every feature pair in every dataset is correlated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["fever", "dependence", "tree", "human", "mnist", "linear", "all"],
                        default="all")
    parser.add_argument("--collect-votes", action="store_true")
    args = parser.parse_args()

    if args.only in ("fever", "all"):
        experiment_fever_cough()
    if args.only in ("dependence", "all"):
        experiment_dependence()
    if args.only in ("tree", "all"):
        experiment_tree_comparison()
    if args.only in ("human", "all"):
        experiment_human_allocation(collect_votes=args.collect_votes)
    if args.only in ("mnist", "all"):
        experiment_mnist()
    if args.only in ("linear", "all"):
        experiment_linear()


if __name__ == "__main__":
    main()
