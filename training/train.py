"""Command-line training entry point; training starts only when explicitly invoked."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from models.model_factory import create_model
from preprocessing.data_validation import validate_dataset
from preprocessing.preprocess import preprocess_dataset
from preprocessing.data_loader import load_dataset
from experiments.manager import ExperimentManager
from project_paths import PROJECT_ROOT, resolve_project_path
from .dataset import create_dataloaders
from .trainer import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Seven_Level_Inverter_Project.zip",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    config_path = resolve_project_path(args.config)
    dataset_path = resolve_project_path(args.dataset)
    experiment_directory = (
        resolve_project_path(args.experiment_dir)
        if args.experiment_dir is not None
        else None
    )
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    if args.resume and args.experiment_dir is None:
        raise ValueError("--resume requires --experiment-dir pointing to an existing run.")
    experiment_cfg = config.get("experiment", {})
    experiment = ExperimentManager(
        root=resolve_project_path(experiment_cfg.get("root", "experiments")),
        name=experiment_cfg.get("name", "inverter-cnn"),
        directory=experiment_directory,
    )
    experiment.save_config(config)

    tables = load_dataset(dataset_path)
    validate_dataset(
        tables,
        report_path=resolve_project_path("results/dataset_validation.json"),
    )
    prepared = preprocess_dataset(
        tables,
        metadata_path=resolve_project_path(
            "data/processed/preprocessing_metadata.json"
        ),
    )
    loaders = create_dataloaders(
        prepared,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
        seed=int(config["training"]["seed"]),
    )
    model = create_model(config)
    train_values = {
        key: value for key, value in config["training"].items()
        if key in {field.name for field in fields(TrainConfig)}
    }
    trainer = Trainer(
        model,
        TrainConfig(**train_values),
        checkpoint_path=experiment.checkpoint_dir / "best.pt",
        history_path=experiment.result_dir / "training_history.csv",
        checkpoint_metadata={
            "model_config": config["model"],
            "normalization": prepared.metadata()["normalization"],
            "operating_feature_names": prepared.metadata()["operating_feature_names"],
        },
    )
    trainer.fit(loaders.train, loaders.validation, resume=args.resume)


if __name__ == "__main__":
    main()
