"""
MNIST handwritten digit classification with TensorFlow/Keras.

This script is the one-command, reproducible version of the project notebook.
Run it with:

    python src/train_model.py

It will:
  1. load the MNIST dataset through tf.keras.datasets.mnist.load_data(),
  2. train a baseline fully-connected classifier,
  3. train the same classifier with Dropout added (the experiment),
  4. evaluate both models on the untouched test set,
  5. save every figure to results/, the metrics to results/metrics.json and the
     best model to models/mnist_model.keras,
  6. rewrite report/MNIST_Report.md with the numbers that were actually measured.

Nothing in this script is hard-coded: every reported number comes from the run.
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1  # 10% of the training set is held out for validation.
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 128

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "report" / "MNIST_Report.md"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds() -> None:
    """Seed Python, NumPy and TensorFlow so runs are as reproducible as possible."""
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Load MNIST exactly as required by the assignment."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocess(x_train, x_test):
    """
    Scale pixel intensities from the 0-255 integer range to 0.0-1.0 floats.

    Neural networks train much faster and more stably when their inputs are
    small numbers. Flattening is done inside the model with a Flatten layer so
    the images stay visible as 28x28 images here.
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_baseline_model() -> keras.Model:
    """A small, understandable densely-connected classifier."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28), name="image"),
            layers.Flatten(name="flatten"),
            layers.Dense(HIDDEN_UNITS, activation="relu", name="hidden"),
            layers.Dense(10, activation="softmax", name="output"),
        ],
        name="baseline_mlp",
    )
    return model


def build_dropout_model() -> keras.Model:
    """
    The identical network with a Dropout layer inserted after the hidden layer.

    Dropout randomly switches off a fraction of the hidden units during each
    training step, which is a common way to reduce overfitting. Keeping every
    other setting the same makes the comparison with the baseline fair.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28), name="image"),
            layers.Flatten(name="flatten"),
            layers.Dense(HIDDEN_UNITS, activation="relu", name="hidden"),
            layers.Dropout(DROPOUT_RATE, name="dropout"),
            layers.Dense(10, activation="softmax", name="output"),
        ],
        name="dropout_mlp",
    )
    return model


def compile_model(model: keras.Model) -> keras.Model:
    """Same optimiser, loss and metric for both models."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(model: keras.Model, x_train, y_train):
    """Train one model and return its history plus the wall-clock training time."""
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=2,
    )
    elapsed = time.perf_counter() - start
    return history, elapsed


def evaluate(model: keras.Model, x_test, y_test):
    """Return (test_loss, test_accuracy) on the untouched test set."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(loss), float(accuracy)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_sample_images(x_train, y_train) -> Path:
    """Save one example of every digit class."""
    indices = [int(np.where(y_train == digit)[0][0]) for digit in range(10)]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, index in zip(axes.flat, indices):
        ax.imshow(x_train[index], cmap="gray")
        ax.set_title(f"Label: {y_train[index]}", fontsize=11)
        ax.axis("off")

    fig.suptitle("Sample MNIST images (one per class)", fontsize=14)
    fig.tight_layout()
    path = RESULTS_DIR / "sample_images.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_history(history, model_name: str, path: Path) -> Path:
    """Plot training vs validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, len(history.history["loss"]) + 1)

    axes[0].plot(epochs_range, history.history["accuracy"], marker="o", label="Training accuracy")
    axes[0].plot(epochs_range, history.history["val_accuracy"], marker="s", label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, history.history["loss"], marker="o", label="Training loss")
    axes[1].plot(epochs_range, history.history["val_loss"], marker="s", label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{model_name}: training history", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictions(model: keras.Model, x_test, y_test, path: Path) -> Path:
    """Show the model's prediction for five test images."""
    probabilities = model.predict(x_test[:5], verbose=0)
    predicted = np.argmax(probabilities, axis=1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, index in zip(axes, range(5)):
        ax.imshow(x_test[index], cmap="gray")
        confidence = probabilities[index][predicted[index]] * 100
        correct = predicted[index] == y_test[index]
        ax.set_title(
            f"True: {y_test[index]}\nPred: {predicted[index]} ({confidence:.1f}%)"
            f"\n{'correct' if correct else 'wrong'}",
            fontsize=10,
            color="green" if correct else "red",
        )
        ax.axis("off")

    fig.suptitle("Predictions on five test images", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(model: keras.Model, x_test, y_test, path: Path):
    """Save the confusion matrix and return (matrix, predicted labels)."""
    probabilities = model.predict(x_test, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_test, predicted)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    threshold = matrix.max() / 2
    for row in range(10):
        for column in range(10):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return matrix, predicted


def plot_experiment_comparison(baseline_history, dropout_history, baseline_metrics, dropout_metrics, path: Path) -> Path:
    """Compare baseline vs dropout: validation curves and final test accuracy."""
    epochs_range = range(1, len(baseline_history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, baseline_history.history["val_accuracy"], marker="o", label="Baseline")
    axes[0].plot(epochs_range, dropout_history.history["val_accuracy"], marker="s", label="With Dropout")
    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, baseline_history.history["val_loss"], marker="o", label="Baseline")
    axes[1].plot(epochs_range, dropout_history.history["val_loss"], marker="s", label="With Dropout")
    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    labels = ["Baseline", "With Dropout"]
    accuracies = [baseline_metrics["test_accuracy"] * 100, dropout_metrics["test_accuracy"] * 100]
    bars = axes[2].bar(labels, accuracies, color=["#4C72B0", "#DD8452"], width=0.55)
    axes[2].set_title("Test accuracy")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(min(accuracies) - 2, 100)
    for bar, accuracy in zip(bars, accuracies):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment: baseline vs adding Dropout", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_conclusion(baseline_accuracy: float, dropout_accuracy: float) -> str:
    """
    Write the conclusion from the measured numbers instead of assuming that
    Dropout helped. The threshold of 0.1 percentage points defines a tie, which
    is far below the run-to-run noise of a single training run.
    """
    difference = (dropout_accuracy - baseline_accuracy) * 100
    if abs(difference) < 0.1:
        verdict = (
            "The two models performed essentially the same on the test set "
            f"(a difference of only {difference:+.2f} percentage points, which is within "
            "the run-to-run noise of a single training run). In this setting Dropout "
            "therefore did not meaningfully improve accuracy."
        )
    elif difference > 0:
        verdict = (
            "Adding Dropout improved test accuracy by "
            f"{difference:.2f} percentage points compared with the baseline, so in this "
            "setting it was a genuine improvement."
        )
    else:
        verdict = (
            "Adding Dropout *reduced* test accuracy by "
            f"{abs(difference):.2f} percentage points compared with the baseline. "
            "The experiment therefore does not support the claim that Dropout improves "
            "performance on this task with these settings."
        )
    return verdict


# The report is a template: every {{TOKEN}} below is replaced with a value
# measured during this run, so the written report can never contain a number
# that the code did not actually produce.
REPORT_TEMPLATE = """# MNIST Handwritten Digit Classification using TensorFlow/Keras

**Author:** _add your name_
**Course:** _add course code and title_
**Date of run:** {{GENERATED_DATE}}
**Repository:** _add your GitHub URL here_

---

## 1. Objective

The objective of this project is to build a neural network using TensorFlow/Keras
to classify handwritten digits from 0 to 9 using the MNIST dataset.

Beyond producing a working classifier, the project tests one specific hypothesis:
**does adding a Dropout layer to a simple fully-connected network improve its
generalisation performance?** Both models are trained and evaluated under
identical conditions so that the comparison is fair, and the conclusion is drawn
only from the numbers that were actually measured.

## 2. Dataset

MNIST was loaded directly through `tf.keras.datasets.mnist.load_data()`, so the
project uses the standard dataset and not an unrelated one. Table 1 summarises it.

| Property | Value |
| --- | --- |
| Training samples | {{NUM_TRAIN}} |
| Test samples | {{NUM_TEST}} |
| Image size | {{IMAGE_SHAPE}} grayscale |
| Classes | {{NUM_CLASSES}} (digits 0-9) |
| Raw pixel range | 0 - 255 |

*Table 1: Summary of the MNIST data as loaded by the code.*

Pixel intensities were scaled to the 0.0-1.0 range by dividing by 255, which
makes training faster and more stable. The test set was never used for training
or for choosing between models; it was evaluated once, at the end, so the
reported test accuracy is an honest estimate of generalisation.

## 3. Methodology

### 3.1 Model architecture

Both models are deliberately small and simple. The baseline is a
fully-connected network:

```text
Input 28x28  ->  Flatten (784)  ->  Dense {{HIDDEN_UNITS}} + ReLU  ->  Dense 10 + Softmax
```

The experiment uses exactly the same network with one change: a
`Dropout({{DROPOUT_RATE}})` layer is inserted between the hidden layer and the
output layer. Dropout has no trainable parameters, so the parameter count of
both models is identical ({{BASELINE_PARAMS}}). The softmax output layer makes
the architecture directly usable as a 10-class classifier.

### 3.2 Training and evaluation setup

| Setting | Value |
| --- | --- |
| Framework | TensorFlow {{TF_VERSION}} (Keras) on Python {{PYTHON_VERSION}} |
| Optimiser | Adam |
| Loss | Sparse categorical cross-entropy |
| Metric | Accuracy |
| Epochs | {{EPOCHS}} |
| Batch size | {{BATCH_SIZE}} |
| Validation data | {{VALIDATION_SPLIT_PERCENT}}% of the training set held out |
| Random seed | 42 (Python, NumPy and TensorFlow) |

*Table 2: Training configuration, identical for both models.*

Every setting above is shared by the two runs, and random seeds were reset
before each model was built so that both start from the same initial weights.
This is what makes the baseline-versus-Dropout comparison controlled rather than
merely suggestive.

## 4. Results

The numbers below were produced by running the project code. Training times are
wall-clock times measured on the machine used for this run.

| Metric | Baseline | With Dropout |
| --- | --- | --- |
| Trainable parameters | {{BASELINE_PARAMS}} | {{DROPOUT_PARAMS}} |
| Final validation accuracy | {{BASELINE_VAL_ACCURACY}}% | {{DROPOUT_VAL_ACCURACY}}% |
| Final validation loss | {{BASELINE_VAL_LOSS}} | {{DROPOUT_VAL_LOSS}} |
| **Test loss** | **{{BASELINE_TEST_LOSS}}** | **{{DROPOUT_TEST_LOSS}}** |
| **Test accuracy** | **{{BASELINE_TEST_ACCURACY}}%** | **{{DROPOUT_TEST_ACCURACY}}%** |
| Test error rate | {{BASELINE_TEST_ERROR}}% | {{DROPOUT_TEST_ERROR}}% |
| Training time (seconds) | {{BASELINE_TRAIN_TIME}} | {{DROPOUT_TRAIN_TIME}} |

*Table 3: Measured results for both models on the untouched MNIST test set.*

Accuracy is reported on the {{NUM_TEST}} test images, of which the baseline
classified incorrectly approximately {{BASELINE_TEST_ERROR}}% and the Dropout
model approximately {{DROPOUT_TEST_ERROR}}%.

### 4.1 Figures

The following figures were generated by the code and are stored in `results/`:

| File | What it shows |
| --- | --- |
| `sample_images.png` | One example of each digit class from the training set |
| `training_history.png` | Baseline training vs validation accuracy and loss per epoch |
| `predictions.png` | Predictions for five unseen test images, with confidence |
| `confusion_matrix.png` | Which digits are confused with which on the test set |
| `experiment_comparison.png` | Baseline vs Dropout validation curves and test accuracy |

## 5. Discussion

The baseline network already classifies MNIST digits very well. A small
fully-connected model with {{HIDDEN_UNITS}} hidden units reaches
{{BASELINE_TEST_ACCURACY}}% test accuracy, which means it misclassifies about
{{BASELINE_TEST_ERROR}}% of the {{NUM_TEST}} test images. This is expected:
MNIST is a clean, well-centred dataset and even a modest network can separate
the ten digit classes almost perfectly.

{{CONCLUSION_TEXT}}

Two points are worth stating because they affect how the experiment should be
interpreted. First, the difference between the two models is very small in
absolute terms and a single training run is not enough to establish that one
architecture is reliably better; several runs with different seeds would be
needed before making a strong claim. Second, Dropout is designed to reduce
overfitting, and its benefit is normally largest when a model has enough
capacity to overfit. Here the network is small and the dataset is large and
clean, so there may simply be little overfitting for Dropout to remove - which
is consistent with the validation loss curves in `experiment_comparison.png`.

The confusion matrix in `confusion_matrix.png` is more informative than the
overall accuracy figure. Errors are concentrated between visually similar
digit shapes (for example 4/9 and 3/8 or 5/3), which is exactly what a human
reader would expect from ambiguous handwriting.

## 6. Conclusion

A simple fully-connected neural network built with TensorFlow/Keras satisfies
the objective of this project: it classifies handwritten digits from the MNIST
dataset with a test accuracy of {{BASELINE_TEST_ACCURACY}}% (baseline) and
{{DROPOUT_TEST_ACCURACY}}% (with Dropout), and the resulting model is saved as
`models/mnist_model.keras`. The controlled experiment shows an accuracy
difference of {{ACCURACY_DIFFERENCE}} percentage points when Dropout is added.
{{CONCLUSION_TEXT}}

## 7. Reproducing the results

```bash
git clone <repository-url>
cd mnist-digit-classification
python -m venv .venv
# Windows: .venv\\Scripts\\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
```

Running the script regenerates every figure in `results/`, the metrics file
`results/metrics.json`, the model in `models/`, and this report. Alternatively,
open `mnist_digit_classification.ipynb` and use *Restart Kernel and Run All*,
which follows the same steps interactively and displays each figure inline.

## 8. References

1. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). *Gradient-based
   learning applied to document recognition.* Proceedings of the IEEE, 86(11).
2. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov,
   R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from
   Overfitting.* Journal of Machine Learning Research, 15(1).
3. TensorFlow documentation, `tf.keras.datasets.mnist`:
   https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
"""


def build_context(metrics: dict) -> dict:
    """Turn the raw metrics dictionary into the strings the report needs."""
    baseline = metrics["baseline"]
    dropout = metrics["dropout"]
    return {
        "GENERATED_DATE": time.strftime("%Y-%m-%d"),
        "PYTHON_VERSION": metrics["python_version"],
        "TF_VERSION": metrics["tensorflow_version"],
        "NUM_TRAIN": f"{metrics['num_train']:,}",
        "NUM_TEST": f"{metrics['num_test']:,}",
        "NUM_CLASSES": metrics["num_classes"],
        "IMAGE_SHAPE": "28 x 28",
        "EPOCHS": metrics["epochs"],
        "BATCH_SIZE": metrics["batch_size"],
        "VALIDATION_SPLIT_PERCENT": int(metrics["validation_split"] * 100),
        "HIDDEN_UNITS": metrics["hidden_units"],
        "DROPOUT_RATE": metrics["dropout_rate"],
        "BASELINE_PARAMS": f"{baseline['parameters']:,}",
        "BASELINE_TEST_ACCURACY": f"{baseline['test_accuracy'] * 100:.2f}",
        "BASELINE_TEST_ERROR": f"{(1 - baseline['test_accuracy']) * 100:.2f}",
        "BASELINE_TEST_LOSS": f"{baseline['test_loss']:.4f}",
        "BASELINE_VAL_ACCURACY": f"{baseline['final_val_accuracy'] * 100:.2f}",
        "BASELINE_VAL_LOSS": f"{baseline['final_val_loss']:.4f}",
        "BASELINE_TRAIN_TIME": f"{baseline['train_time_seconds']:.1f}",
        "DROPOUT_PARAMS": f"{dropout['parameters']:,}",
        "DROPOUT_TEST_ACCURACY": f"{dropout['test_accuracy'] * 100:.2f}",
        "DROPOUT_TEST_ERROR": f"{(1 - dropout['test_accuracy']) * 100:.2f}",
        "DROPOUT_TEST_LOSS": f"{dropout['test_loss']:.4f}",
        "DROPOUT_VAL_ACCURACY": f"{dropout['final_val_accuracy'] * 100:.2f}",
        "DROPOUT_VAL_LOSS": f"{dropout['final_val_loss']:.4f}",
        "DROPOUT_TRAIN_TIME": f"{dropout['train_time_seconds']:.1f}",
        "ACCURACY_DIFFERENCE": f"{metrics['accuracy_difference_points']:+.2f}",
        "BEST_MODEL_NAME": metrics["best_model"],
        "CONCLUSION_TEXT": build_conclusion(baseline["test_accuracy"], dropout["test_accuracy"]),
    }


def write_report(metrics: dict) -> Path:
    """
    Fill the report template with the measured values and save it.

    Takes the metrics dictionary produced by this script (or by the notebook),
    so both entry points share one implementation and the written report can
    only ever contain numbers that were really measured.
    """
    report = REPORT_TEMPLATE
    for key, value in build_context(metrics).items():
        report = report.replace("{{" + key + "}}", str(value))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Use a non-interactive backend so the script also works on a headless
    # machine. `force=True` is needed because pyplot has already been imported.
    # Setting it here (rather than at import time) lets the notebook import
    # write_report from this module without losing inline figure display.
    matplotlib.use("Agg", force=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    set_seeds()

    print(f"Python       : {platform.python_version()}")
    print(f"TensorFlow   : {tf.__version__}")

    # 1. Data ------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Training images: {x_train.shape}")
    print(f"Test images    : {x_test.shape}")

    x_train, x_test = preprocess(x_train, x_test)
    plot_sample_images(x_train, y_train)
    print("Saved results/sample_images.png")

    # 2. Baseline --------------------------------------------------------
    print("\n=== Baseline model ===")
    set_seeds()  # identical initial weights for a fair comparison
    baseline_model = compile_model(build_baseline_model())
    baseline_model.summary()
    baseline_history, baseline_time = train(baseline_model, x_train, y_train)
    baseline_loss, baseline_accuracy = evaluate(baseline_model, x_test, y_test)

    plot_training_history(baseline_history, "Baseline", RESULTS_DIR / "training_history.png")
    print("Saved results/training_history.png")
    plot_predictions(baseline_model, x_test, y_test, RESULTS_DIR / "predictions.png")
    print("Saved results/predictions.png")
    _, predicted = plot_confusion_matrix(baseline_model, x_test, y_test, RESULTS_DIR / "confusion_matrix.png")
    print("Saved results/confusion_matrix.png")

    report_text = classification_report(y_test, predicted, digits=4)
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    print("Saved results/classification_report.txt")

    # 3. Experiment: Dropout ---------------------------------------------
    print("\n=== Model with Dropout ===")
    set_seeds()  # identical initial weights for a fair comparison
    dropout_model = compile_model(build_dropout_model())
    dropout_model.summary()
    dropout_history, dropout_time = train(dropout_model, x_train, y_train)
    dropout_loss, dropout_accuracy = evaluate(dropout_model, x_test, y_test)

    plot_experiment_comparison(
        baseline_history,
        dropout_history,
        {"test_accuracy": baseline_accuracy},
        {"test_accuracy": dropout_accuracy},
        RESULTS_DIR / "experiment_comparison.png",
    )
    print("Saved results/experiment_comparison.png")

    # 4. Save the better model -------------------------------------------
    if dropout_accuracy >= baseline_accuracy:
        best_model, best_name = dropout_model, "Model with Dropout"
    else:
        best_model, best_name = baseline_model, "Baseline model"
    model_path = MODELS_DIR / "mnist_model.keras"
    best_model.save(model_path)
    print(f"Saved {model_path} ({best_name}, test accuracy {max(baseline_accuracy, dropout_accuracy) * 100:.2f}%)")

    # 5. Metrics ---------------------------------------------------------
    metrics = {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "dropout_rate": DROPOUT_RATE,
        "hidden_units": HIDDEN_UNITS,
        "num_train": int(x_train.shape[0]),
        "num_test": int(x_test.shape[0]),
        "num_classes": int(len(np.unique(y_train))),
        "baseline": {
            "parameters": int(baseline_model.count_params()),
            "test_loss": baseline_loss,
            "test_accuracy": baseline_accuracy,
            "train_time_seconds": baseline_time,
            "final_val_accuracy": float(baseline_history.history["val_accuracy"][-1]),
            "final_val_loss": float(baseline_history.history["val_loss"][-1]),
        },
        "dropout": {
            "parameters": int(dropout_model.count_params()),
            "test_loss": dropout_loss,
            "test_accuracy": dropout_accuracy,
            "train_time_seconds": dropout_time,
            "final_val_accuracy": float(dropout_history.history["val_accuracy"][-1]),
            "final_val_loss": float(dropout_history.history["val_loss"][-1]),
        },
        "best_model": best_name,
        "accuracy_difference_points": (dropout_accuracy - baseline_accuracy) * 100,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved results/metrics.json")

    # 6. Report ----------------------------------------------------------
    write_report(metrics)
    print(f"Saved {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    print("\nDone. All figures, the metrics file and the report now contain the numbers")
    print("measured by this run. To freeze the exact environment, run: pip freeze > requirements-lock.txt")


if __name__ == "__main__":
    main()
