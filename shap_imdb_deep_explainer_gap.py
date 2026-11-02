#!/usr/bin/env python3
"""
IMDB sentiment classification with word-position explanations using SHAP DeepExplainer.

Architecture:
    Integer word IDs -> Embedding -> GlobalAveragePooling1D -> Dense -> Sigmoid

DeepExplainer is applied to the FLOATING-POINT embedding representations, not
to integer word IDs. Embedding-dimension SHAP values are summed to produce
one signed score per token position.

Install:
    pip install tensorflow shap numpy matplotlib

Run:
    python shap_imdb_deep_explainer.py

Example:
    python shap_imdb_deep_explainer.py --epochs 5 --background-size 100 --num-explain 5
"""

import argparse
import os
import random
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain IMDB sentiment predictions with SHAP DeepExplainer."
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--num-explain", type=int, default=3)
    parser.add_argument("--explain-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="shap_imdb_results")
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(vocab_size, maxlen):
    print("Loading the IMDB movie-review sentiment dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=vocab_size
    )

    x_train = pad_sequences(
        x_train, maxlen=maxlen, padding="post", truncating="post"
    )
    x_test = pad_sequences(
        x_test, maxlen=maxlen, padding="post", truncating="post"
    )

    print(f"Training examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")
    print(f"Padded sequence shape: {x_train.shape}")

    return (
        np.asarray(x_train, dtype=np.int32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(x_test, dtype=np.int32),
        np.asarray(y_test, dtype=np.float32),
    )


def build_model(vocab_size, maxlen, embedding_dim):
    # mask_zero=False is intentional: the same pooling operation is reused
    # by the embedding-input model passed to DeepExplainer.
    inputs = tf.keras.Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="word_embedding",
    )(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling"
    )(embedding)
    output = tf.keras.layers.Dense(
        1, activation="sigmoid", name="positive_probability"
    )(pooled)

    model = tf.keras.Model(inputs, output, name="imdb_average_sentiment")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_embedding_input_model(trained_model, maxlen, embedding_dim):
    """
    Rebuild the trained model from its embedding output onward.

    This model accepts float32 embedding tensors, which are differentiable.
    The original trained layer objects and weights are reused.
    """
    embedding_input = tf.keras.Input(
        shape=(maxlen, embedding_dim),
        dtype=tf.float32,
        name="embedding_input",
    )

    pooling_layer = trained_model.get_layer("global_average_pooling")
    dense_layer = trained_model.get_layer("positive_probability")

    x = pooling_layer(embedding_input)
    output = dense_layer(x)

    return tf.keras.Model(
        inputs=embedding_input,
        outputs=output,
        name="sentiment_model_from_embeddings",
    )


def normalize_shap_values(raw_values):
    """Handle SHAP's list and ndarray return formats across versions."""
    if isinstance(raw_values, list):
        if len(raw_values) != 1:
            raise RuntimeError(
                f"Expected one model output, but SHAP returned {len(raw_values)} outputs."
            )
        values = np.asarray(raw_values[0])
    else:
        values = np.asarray(raw_values)

    # Common shapes:
    # (samples, tokens, embedding_dim, 1) or
    # (samples, tokens, embedding_dim)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    elif values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0, ...]

    if values.ndim != 3:
        raise RuntimeError(
            "Unexpected SHAP array shape "
            f"{values.shape}. Expected (samples, tokens, embedding_dim)."
        )

    return values


def make_word_maps(vocab_size):
    word_index = imdb.get_word_index()
    # Keras IMDB encoding reserves 0 for padding and uses offsets:
    # 0 = padding, 1 = start, 2 = OOV, and word_index + 3 = token ID.
    reverse_index = {index + 3: word for word, index in word_index.items()}
    reverse_index[0] = "<PAD>"
    reverse_index[1] = "<START>"
    reverse_index[2] = "<OOV>"
    return reverse_index


def token_to_word(token_id, reverse_index):
    return reverse_index.get(int(token_id), f"<ID:{int(token_id)}>")


def explain_and_plot(
    model,
    x_test,
    y_test,
    args,
    reverse_index,
):
    rng = np.random.default_rng(args.seed)

    background_count = min(args.background_size, len(x_test))
    if background_count < 1:
        raise ValueError("--background-size must be at least 1.")

    # Randomly sample background reviews from the test set.
    background_indices = rng.choice(
        len(x_test), size=background_count, replace=False
    )
    background_ids = x_test[background_indices]

    start = max(0, args.explain_index)
    end = min(start + args.num_explain, len(x_test))
    if start >= len(x_test):
        raise ValueError("--explain-index is outside the test dataset.")
    if args.num_explain < 1:
        raise ValueError("--num-explain must be at least 1.")

    explain_ids = x_test[start:end]
    explain_labels = y_test[start:end]

    embedding_layer = model.get_layer("word_embedding")
    embedding_model = build_embedding_input_model(
        model, args.maxlen, args.embedding_dim
    )

    # Convert integer token IDs to float32 embedding vectors.
    background_embeddings = embedding_layer(
        tf.convert_to_tensor(background_ids, dtype=tf.int32)
    ).numpy().astype(np.float32)

    explain_embeddings = embedding_layer(
        tf.convert_to_tensor(explain_ids, dtype=tf.int32)
    ).numpy().astype(np.float32)

    print(
        f"\nCreating DeepExplainer with {background_count} background reviews "
        f"and explaining {len(explain_ids)} review(s)..."
    )
    print("DeepExplainer input shape:", explain_embeddings.shape)

    explainer = shap.DeepExplainer(
        embedding_model,
        background_embeddings,
    )

    # check_additivity=False avoids version-specific numerical additivity
    # checks; it does not change the SHAP values being calculated.
    try:
        raw_values = explainer.shap_values(
            explain_embeddings,
            check_additivity=False,
        )
    except TypeError:
        # Compatibility with SHAP versions that do not expose this argument.
        raw_values = explainer.shap_values(explain_embeddings)

    embedding_shap = normalize_shap_values(raw_values)

    # Sum attribution across embedding dimensions to obtain one score per token.
    token_shap = embedding_shap.sum(axis=-1)

    os.makedirs(args.output_dir, exist_ok=True)

    predicted_probabilities = model.predict(
        explain_ids, verbose=0
    ).reshape(-1)

    np.savez_compressed(
        os.path.join(args.output_dir, "imdb_deep_shap_values.npz"),
        embedding_shap_values=embedding_shap,
        token_shap_values=token_shap,
        token_ids=explain_ids,
        true_labels=explain_labels,
        predicted_probabilities=predicted_probabilities,
    )

    for local_index, (token_ids, scores, probability, true_label) in enumerate(
        zip(explain_ids, token_shap, predicted_probabilities, explain_labels)
    ):
        # Exclude padding tokens from the displayed word list.
        valid_positions = np.flatnonzero(token_ids != 0)
        words = [
            token_to_word(token_ids[position], reverse_index)
            for position in valid_positions
        ]
        word_scores = scores[valid_positions]

        if len(words) == 0:
            print(f"Review {start + local_index}: contains no non-padding tokens.")
            continue

        # Display all words in their original sequence order.
        # Positive values push toward positive sentiment; negative values
        # push toward negative sentiment.
        order = np.argsort(word_scores)
        sorted_words = [words[i] for i in order]
        sorted_scores = word_scores[order]

        height = max(4, 0.28 * len(sorted_words))
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(sorted_words, sorted_scores)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("SHAP value (sum across embedding dimensions)")
        ax.set_title(
            f"Review {start + local_index} | "
            f"Predicted positive probability: {probability:.3f} | "
            f"True label: {'positive' if true_label >= 0.5 else 'negative'}"
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        output_path = os.path.join(
            args.output_dir,
            f"imdb_deep_shap_review_{start + local_index}.png",
        )
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved explanation: {output_path}")

        top_positive = np.argsort(word_scores)[-5:][::-1]
        top_negative = np.argsort(word_scores)[:5]

        print("\n  Strongest positive word-position contributions:")
        for index in top_positive:
            print(f"    {words[index]:<20} {word_scores[index]: .5f}")

        print("\n  Strongest negative word-position contributions:")
        for index in top_negative:
            print(f"    {words[index]:<20} {word_scores[index]: .5f}")
        print()

    print(f"Saved numerical SHAP values in: {args.output_dir}/imdb_deep_shap_values.npz")


def main():
    args = parse_args()
    set_seeds(args.seed)

    x_train, y_train, x_test, y_test = load_data(
        args.vocab_size, args.maxlen
    )

    model = build_model(
        args.vocab_size,
        args.maxlen,
        args.embedding_dim,
    )
    model.summary()

    print("\nTraining sentiment model...")
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    reverse_index = make_word_maps(args.vocab_size)

    explain_and_plot(
        model=model,
        x_test=x_test,
        y_test=y_test,
        args=args,
        reverse_index=reverse_index,
    )


if __name__ == "__main__":
    main()
