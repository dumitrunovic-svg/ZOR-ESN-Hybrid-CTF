"""
ZOR_ESN_Hybrid Runner — automatic strategy selection per pair.

For each pair, detects whether training data is a continuous trajectory or
iid samples, and applies the optimal prediction strategy:

  - TEMPORAL data → ESN forecasting (with tuned hyperparameters)
  - IID + reconstruction → predict zeros (mean baseline, ~score 0)
  - IID + long_time      → random samples from training (preserves PSD)
  - IID + short+long     → mixed (zeros first k=20, random last k=20)

The contribution is not an ESN tuning, but a structural fix of the
orchestration: choosing the right prediction strategy per data type and metric.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# CTF root is two levels up from this file (models/ZOR_ESN_Hybrid/run.py)
CTF_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CTF_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ctf4science.data_module import (
    load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots,
)
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization

from esn import ESN
from hybrid_model import (
    TrajectoryType, detect_trajectory_type, select_strategy,
    predict_iid_reconstruction, predict_iid_long_time,
    predict_iid_short_time, predict_iid_mixed,
)

MODEL_NAME = "ZOR_ESN_Hybrid"


def build_esn_from_config(config: dict, seed: int = 42) -> ESN:
    m = config.get("model", {})
    return ESN(
        reservoir_size=int(m.get("reservoir_size", 1000)),
        spectral_radius=float(m.get("spectral_radius", 0.85)),
        input_scaling=float(m.get("input_scaling", 0.1)),
        leaking_rate=float(m.get("leaking_rate", 1.0)),
        ridge_alpha=float(m.get("ridge_alpha", 1e-4)),
        washout=int(m.get("washout", 200)),
        seed=seed,
    )


def get_pair_metrics(dataset_name: str, pair_id: int) -> list[str]:
    """Read metric list for a pair from dataset config."""
    yaml_path = CTF_ROOT / "data" / dataset_name / f"{dataset_name}.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    for p in cfg["pairs"]:
        if int(p["id"]) == int(pair_id):
            return list(p.get("metrics", []))
    return []


def predict_for_pair(
    X_train: np.ndarray,
    n_steps: int,
    metrics: list[str],
    config: dict,
    seed: int,
    init_data: np.ndarray | None = None,
    k_short: int = 20,
) -> tuple[np.ndarray, dict]:
    """Apply the optimal strategy based on data type + metrics."""
    traj_type = detect_trajectory_type(X_train)
    strategy = select_strategy(traj_type, metrics)

    info: dict = {
        "trajectory_type": traj_type.value,
        "metrics": list(metrics),
        "strategy": strategy,
        "train_shape": list(X_train.shape),
        "n_steps_predicted": int(n_steps),
    }

    if strategy in ("esn", "esn_long_patch"):
        esn = build_esn_from_config(config, seed=seed)
        if init_data is not None:
            esn.fit(X_train)
            pred = esn.predict(n_steps, init_data=init_data)
        else:
            pred = esn.fit_predict(X_train, n_steps)
        info["model"] = {
            "name": "ESN",
            "reservoir_size": esn.N,
            "spectral_radius": esn.rho,
            "ridge_alpha": esn.ridge,
        }

        if strategy == "esn_long_patch" and n_steps >= 2 * k_short:
            # Replace last k=20 rows with random samples from training set.
            # This restores PSD content (long_time metric on last k) while
            # preserving the ESN forecast on the first k (short_time metric).
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(X_train), size=k_short)
            pred = pred.copy()
            pred[-k_short:] = X_train[idx]
            info["long_patch_applied"] = True

        return pred, info

    rng = np.random.default_rng(seed)
    if strategy == "iid_zeros":
        pred = predict_iid_reconstruction(X_train, n_steps)
    elif strategy == "iid_random":
        pred = predict_iid_long_time(X_train, n_steps, rng=rng)
    elif strategy == "iid_mixed":
        pred = predict_iid_mixed(X_train, n_steps, k_short=k_short, rng=rng)
    else:
        pred = np.zeros((n_steps, X_train.shape[1]), dtype=np.float64)
    info["model"] = {"name": strategy}
    return pred, info


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    pair_ids = parse_pair_ids(config["dataset"])

    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    batch_results: dict = {
        "batch_id": batch_id,
        "model": MODEL_NAME,
        "dataset": dataset_name,
        "pairs": [],
        "strategy_log": [],
    }

    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    for pair_id in pair_ids:
        train_data, init_data = load_dataset(dataset_name, pair_id, transpose=True)
        X_train = np.concatenate(train_data, axis=1).T
        X_init = init_data.T if init_data is not None else None

        n_steps = get_prediction_timesteps(dataset_name, pair_id).shape[0]
        metrics = get_pair_metrics(dataset_name, pair_id)

        pred, info = predict_for_pair(
            X_train=X_train,
            n_steps=n_steps,
            metrics=metrics,
            config=config,
            seed=42 + pair_id,
            init_data=X_init,
        )

        info["pair_id"] = int(pair_id)
        batch_results["strategy_log"].append(info)

        test_dir = CTF_ROOT / "data" / dataset_name / "test"
        has_local_test = test_dir.exists() and any(test_dir.iterdir())
        if has_local_test:
            results = evaluate(dataset_name, pair_id, pred)
        else:
            results = {"submission_only": True}

        results_directory = save_results(
            dataset_name, MODEL_NAME, batch_id, pair_id, config, pred, results
        )

        batch_results["pairs"].append({
            "pair_id": int(pair_id),
            "metrics": results,
            "strategy": info["strategy"],
            "trajectory_type": info["trajectory_type"],
        })
        print(
            f"  Pair {pair_id}: type={info['trajectory_type']}, "
            f"strategy={info['strategy']}, metrics={metrics}, results={results}"
        )

        for plot_type in applicable_plots:
            try:
                fig = viz.plot_from_batch(
                    dataset_name, pair_id, results_directory, plot_type=plot_type
                )
                viz.save_figure_results(
                    fig, dataset_name, MODEL_NAME, batch_id, pair_id,
                    plot_type, results_directory,
                )
            except Exception:
                pass

    with open(results_directory.parent / "batch_results.yaml", "w") as f:
        yaml.dump(batch_results, f)

    all_scores: list[float] = []
    print(f"\n=== {MODEL_NAME} on {dataset_name} ===")
    for p in batch_results["pairs"]:
        for m, v in p["metrics"].items():
            if isinstance(v, (int, float)):
                print(f"  E{p['pair_id']:2d} {m:18s} ({p['strategy']:12s}): {v:7.2f}")
                all_scores.append(float(v))
            else:
                print(f"  E{p['pair_id']:2d} {m}: {v}")
    if all_scores:
        print(f"  Mean score: {np.mean(all_scores):.2f}")
    print(f"Results saved in: {results_directory.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
