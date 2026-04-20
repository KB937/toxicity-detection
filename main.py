"""
main.py — Toxicity Detection Pipeline CLI

This is the orchestration script for the entire NLP research project.
It loads data, preprocesses it, trains a TF-IDF baseline, optionally
fine-tunes a DistilBERT model, computes all metrics, generates all
plots, and performs error analysis.

Usage:
    python main.py --data_path data/train.csv [--sample 20000] [--skip_transformer]
"""

import os
import sys

if sys.version_info >= (3, 13):
    raise RuntimeError(
        "This project requires Python 3.10. "
        "Please create a conda environment with Python 3.10."
    )

import argparse
import logging
import datetime

import pandas as pd

from src.preprocess import CONFIG as PREPROCESS_CONFIG
from src.preprocess import load_data, preprocess_pipeline
from src.baseline_model import build_tfidf_pipeline, train_baseline, predict_baseline, get_top_features
from src.transformer_model import set_all_seeds, train_transformer, predict_transformer
from src.transformer_model import CONFIG as TRANSFORMER_CONFIG
from src.evaluate import (
    compute_metrics,
    print_comparison_table,
    plot_class_distribution,
    plot_wordclouds,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_top_features as plot_top_features_chart,
    show_misclassifications,
)

# ── 1. Parse Arguments & Configure Logging ──────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline run."""
    parser = argparse.ArgumentParser(
        description="Run the toxicity detection pipeline (Baseline + DistilBERT)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Jigsaw train.csv file. Must include 'comment_text' and 'toxic' columns.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="Number of rows to sample from the full dataset (default: 20000).",
    )
    parser.add_argument(
        "--skip_transformer",
        action="store_true",
        help="If set, skip the DistilBERT fine-tuning and just run the TF-IDF baseline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Base directory for saving plots, metrics, and models.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Set up the root logger to output to console with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── 2. Orchestration ────────────────────────────────────────────────

def main():
    args = parse_args()
    configure_logging()
    logger = logging.getLogger("main")

    logger.info("=== Toxicity Detection Pipeline Started ===")
    
    # Ensure full reproducibility across all modules before we load any data
    set_all_seeds(PREPROCESS_CONFIG["random_state"])

    # ── Directory setup ─────────────────────────────────────────────
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "model"), exist_ok=True)

    metrics_list = []

    # ── Data Loading & Preprocessing ────────────────────────────────
    try:
        df_full = load_data(
            filepath=args.data_path,
            sample_n=args.sample,
            random_state=PREPROCESS_CONFIG["random_state"],
        )
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Visualise raw class balance before arbitrary text manipulation
    plot_class_distribution(
        y=df_full[PREPROCESS_CONFIG["target_col"]].values,
        save_path=os.path.join(figures_dir, "class_distribution.png")
    )

    X_train, X_test, y_train, y_test = preprocess_pipeline(df_full)

    # Word clouds (reconstruct a DataFrame of clean strings for plot_wordclouds)
    df_clean = pd.DataFrame({
        PREPROCESS_CONFIG["text_col"]: X_train,
        PREPROCESS_CONFIG["target_col"]: y_train,
    })
    plot_wordclouds(
        df_clean,
        text_col=PREPROCESS_CONFIG["text_col"],
        label_col=PREPROCESS_CONFIG["target_col"],
        save_dir=figures_dir,
    )

    # ── Baseline Model ──────────────────────────────────────────────
    logger.info("--- Starting Baseline Pipeline ---")
    pipeline = train_baseline(X_train, y_train)
    y_pred_b, y_proba_b = predict_baseline(pipeline, X_test)

    metrics_b = compute_metrics(
        y_test, y_pred_b, y_proba_b,
        model_name="TF-IDF + Logistic Reg"
    )
    metrics_list.append(metrics_b)

    plot_confusion_matrix(
        y_test, y_pred_b,
        model_name="TF-IDF + Logistic Reg",
        save_path=os.path.join(figures_dir, "confusion_baseline.png")
    )

    top_features_df = get_top_features(pipeline, n=20)
    plot_top_features_chart(
        top_features_df,
        save_path=os.path.join(figures_dir, "top_features.png")
    )

    # ── Transformer Model ───────────────────────────────────────────
    y_proba_t = None
    y_pred_t = None
    
    if not args.skip_transformer:
        logger.info("--- Starting DistilBERT Pipeline ---")
        train_transformer(X_train, y_train, X_test, y_test)
        
        y_pred_t, y_proba_t = predict_transformer(
            X_test,
            model_dir=TRANSFORMER_CONFIG["output_dir"]
        )
        
        metrics_t = compute_metrics(
            y_test, y_pred_t, y_proba_t,
            model_name="DistilBERT (fine-tuned)"
        )
        metrics_list.append(metrics_t)
        
        plot_confusion_matrix(
            y_test, y_pred_t,
            model_name="DistilBERT",
            save_path=os.path.join(figures_dir, "confusion_transformer.png")
        )
    else:
        logger.info("Skipping DistilBERT pipeline as requested (--skip_transformer).")

    # ── Final Comparative Visualisation & Output ────────────────────
    
    if not args.skip_transformer:
        plot_roc_curves(
            y_test,
            y_proba_b,
            proba_transformer=y_proba_t,
            save_path=os.path.join(figures_dir, "roc_curves.png")
        )
    else:
        logger.info("Transformer not run — ROC curve shows baseline only. Re-run without --skip_transformer to generate full comparison.")
        plot_roc_curves(
            y_test,
            y_proba_b,
            proba_transformer=None,
            save_path=os.path.join(figures_dir, "roc_curves_baseline_only.png")
        )

    # 2. Print metrics table to console
    print_comparison_table(metrics_list)

    # 3. Print error analysis for best model (DistilBERT if run, else Baseline)
    best_pred = y_pred_t if not args.skip_transformer else y_pred_b
    best_name = "DistilBERT" if not args.skip_transformer else "Baseline"
    show_misclassifications(
        X_test,
        y_test,
        best_pred,
        model_name=best_name,
        save_path=os.path.join(args.output_dir, "misclassifications.csv"),
        n=5
    )

    # 4. Final summary block
    print("\n" + "═" * 60)
    print("PIPELINE COMPLETE")
    print(f"Figures saved to: {figures_dir}")
    print(f"Model comparison: {os.path.join(args.output_dir, 'model_comparison.csv')}")
    print(f"Misclassifications: {os.path.join(args.output_dir, 'misclassifications.csv')}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
