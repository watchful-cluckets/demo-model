import argparse
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import seaborn as sns
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define constants
CONFIG_FILE = "feature_config.yaml"
FIGURES_DIR = "figures"
MODELS_DIR = "models"

# start a logger
start_time = time.time()
logfile = os.path.splitext(os.path.basename(__file__))[0] + ".log"
msgfmt = "%(asctime)s %(name)-22s %(levelname)-8s %(message)s"
dtefmt = "%d-%m-%Y %H:%M:%S"
logging.basicConfig(
    level=logging.DEBUG, format=msgfmt, datefmt=dtefmt, filename=logfile, filemode="w"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(msgfmt)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
log = logging.getLogger()


# Feature categories for easier reference and less hard-coding
class Feature(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


def valid_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"The file '{path}' does not exist.")
    return path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predictive model for colonoscopy site selection."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators for RandomForest.",
    )
    parser.add_argument(
        "--max_depth", type=int, default=None, help="Maximum depth for RandomForest."
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum samples split for RandomForest.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3, help="Test size for train-test split."
    )
    parser.add_argument(
        "--datafile",
        type=valid_file,
        default=None,
        help="Path to the data file.",
    )
    return parser.parse_args()


def setup_directories(dirs: List[str]) -> None:
    """
    Ensure that the specified directories exist; create them if they don't.
    """
    for new_dir in dirs:
        os.makedirs(new_dir, exist_ok=True)


def load_config(file_path: str) -> Dict[str, Union[List[str], str]]:
    full_path = os.path.abspath(file_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def load_and_process_data(
    feature_config: Dict[str, Union[List[str], str]],
    test_size: float,
    data_path: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    ColumnTransformer,
    List[str],
    List[str],
]:
    """
    To improve dataset analysis, handle outliers and use an imputation strategy
    for missing data. Add new features like interaction terms and polynomial features
    to boost model performance. Conduct a feature importance analysis to reduce
    dimensionality and speed up training for better overall performance.

    :param feature_config: yaml file with listing of numerical and categorical feature names
    :param test_size: size of the test split in decimal format, e.g., 0.2
    :param data_path: path to the data file containing feature data
    :return:
    """
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_parquet(data_path)
        except pyarrow.ArrowInvalid as e:
            log.error(f"Invalid Parquet file: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred while reading the Parquet file: {e}")
            raise
    else:
        n_sim = 1000
        data = {
            "Age": np.random.randint(40, 80, n_sim),
            "Gender": np.random.choice(["M", "F"], n_sim),
            "Family_History": np.random.choice(["Yes", "No"], n_sim),
            "Previous_Polyps": np.random.choice(["Yes", "No"], n_sim),
            "Symptoms": np.random.choice(["Yes", "No"], n_sim),
            "Lifestyle_Risk": np.random.choice(["High", "Medium", "Low"], n_sim),
            "Other_Conditions": np.random.choice(["Yes", "No"], n_sim),
            "Physician_Recommendation": np.random.choice(["Yes", "No"], n_sim),
            "Outcome": np.random.choice([0, 1], n_sim),
        }
        df = pd.DataFrame(data)
    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    numerical_features = feature_config[Feature.NUMERICAL.value]
    categorical_features = feature_config[Feature.CATEGORICAL.value]

    # Build transformers
    preprocessor = ColumnTransformer(
        transformers=[
            (Feature.NUMERICAL.value, StandardScaler(), numerical_features),
            (
                Feature.CATEGORICAL.value,
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        numerical_features,
        categorical_features,
    )


def build_fit_model(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
) -> GridSearchCV:
    """
    Perform grid search with the given pipeline and hyperparameters.

    :param preprocessor: transformer(s) for feature data
    :param X_train: training feature set
    :param y_train: training label set
    :param args: command line args defining grid params
    :return: fitted GridSearchCV object
    """
    pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("ros", RandomOverSampler(random_state=42)),
            ("classifier", RandomForestClassifier()),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [args.n_estimators],
        "classifier__max_depth": [args.max_depth],
        "classifier__min_samples_split": [args.min_samples_split],
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    log.info(f"Best Parameters: {grid_search.best_params_}")
    log.info(f"Best Score: {grid_search.best_score_}")

    return grid_search


def save_model(model, model_path: str) -> None:
    """
    Save the model to the specified path.

    :param model: The trained machine learning model
    :param model_path: The path where to save the model
    """
    dump(model, model_path)
    log.info(f"Model saved at {model_path}")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: List[str],
    cat_features: List[str],
    figure_path: str,
) -> None:
    """
    Evaluate the model and print various metrics.

    :param model: The trained machine learning model
    :param X_test: The testing feature set
    :param y_test: The testing label set
    :param num_features: names of numerical features
    :param cat_features: names of categorical features
    :param figure_path: The path where to save figures
    """
    # get predicitions for best model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Classification Report
    # TODO: Print performance data to metadata file
    clf_report = classification_report(
        y_test, y_pred, target_names=["Preferred", "Non-preferred"]
    )
    log.info(f"Classification Report:\n{clf_report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, figure_path)

    # Accuracy
    # TODO -- add bootstrap estimate for accuracy ci
    accuracy = accuracy_score(y_test, y_pred)
    # acc_lower, acc_upper = bootstrap_acc_ci()
    # auc_lower, auc_upper = bootstrap_auc_ci()
    log.info(f"Accuracy Score: {accuracy}")
    # log.info(f"Accuracy 95% CI: ({acc_lower:.2f}, {acc_upper:.2f})")

    # ROC-AUC
    # TODO -- add bootstrap estimate for auc
    auc = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plot_roc_auc(fpr, tpr, auc, figure_path)
    log.info(f"ROC-AUC Score: {auc}")
    # log.info(f"ROC-AUC 95% CI: ({auc_lower:.2f}, {auc_upper:.2f})")

    # Feature importance
    clf = model.named_steps["classifier"]
    one_hot_columns = (
        model.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .get_feature_names_out(cat_features)
    )
    feature_names = num_features + list(one_hot_columns)
    plot_feature_importance(
        clf.feature_importances_, feature_names, "Random Forest", figure_path
    )

    # Precision and recall
    avg_precision = average_precision_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plot_precision_recall(precision, recall, avg_precision, figure_path)
    log.info(f"Avg. Precision Score: {avg_precision}")


def plot_roc_auc(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str) -> None:
    """
    Plot and save ROC-AUC curve.

    Parameters:
    - fpr: False positive rates
    - tpr: True positive rates
    - auc: Area under ROC curve
    - save_path: Directory path to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, "roc_auc_curve.png"))
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """
    Plot and save confusion matrix.

    Parameters:
    - cm: Confusion matrix
    - path: Directory path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Preferred", "Non-preferred"],
        yticklabels=["Preferred", "Non-preferred"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


def plot_precision_recall(
    precision: np.ndarray, recall: np.ndarray, average_precision: float, save_path: str
) -> None:
    """
    Plot and save precision-recall curve.

    Parameters:
    - precision: Precision values
    - recall: Recall values
    - average_precision: Average precision score
    - path: Directory path to save the figure
    """
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall curve: AP={average_precision:0.2f}")
    plt.savefig(os.path.join(save_path, "precision_recall_curve.png"))
    plt.close()


def plot_feature_importance(
    importance: np.ndarray, names: List[str], model_type: str, save_path: str
) -> None:
    """
    Plot feature importances for a fitted model.

    Parameters:
        importance (array): Feature importances as returned by the fitted model.
        names (list): List of feature names.
        model_type (str): Type of model (used in plot title).
        save_path (str): Where to save the generated plot.

    Returns:
        None
    """
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}

    # Sort the DataFrame in order decreasing feature importance
    feature_df = pd.DataFrame(data).sort_values("feature_importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(
        y=feature_df["feature_names"],
        width=feature_df["feature_importance"],
        color="skyblue",
    )
    plt.title(model_type + " - Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.savefig(os.path.join(save_path, "feature_importance.png"))
    plt.close()


def main():
    """Main function to run the entire pipeline."""
    args = parse_arguments()
    setup_directories([FIGURES_DIR, MODELS_DIR])
    feature_config = load_config(CONFIG_FILE)

    # Data Preprocessing and Exploration
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        num_features,
        cat_features,
    ) = load_and_process_data(
        feature_config=feature_config, test_size=args.test_size, data_path=args.datafile
    )

    # Create an imbalanced-learn pipeline
    grid_search_models = build_fit_model(preprocessor, X_train, y_train, args)
    best_model = grid_search_models.best_estimator_
    save_model(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    # Evaluate Model
    evaluate_model(best_model, X_test, y_test, num_features, cat_features, FIGURES_DIR)


if __name__ == "__main__":
    main()
