"""
CLI — Command-line interface for multisync-core.

Usage:
    python -m multisync analyze -i neural.csv,bio.csv,behavior.csv \
           -n neural,bio,behavior --hz 1.0 -o results.json

    python -m multisync demo --output demo_results.json
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .core import Dyad, DynamicAnalyzer
from .dataset import SynchronyDataset
from .io import load_csv
from .synthetic import generate_ground_truth_dyad


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run analysis on user-provided CSV files."""
    input_files = args.input.split(",")
    names = args.names.split(",") if args.names else [
        f"modality_{i}" for i in range(len(input_files))
    ]

    if len(input_files) != len(names):
        print("Error: number of input files must match number of names.", file=sys.stderr)
        sys.exit(1)

    hz = float(args.hz)
    modalities = {}
    for name, path in zip(names, input_files):
        print(f"  Loading {name}: {path}")
        modalities[name] = load_csv(path)

    # Create dyad and run pipeline
    dyad = Dyad(**modalities, hz=hz)

    # Add context labels if provided
    if args.contexts:
        for ctx_str in args.contexts:
            parts = ctx_str.split(",")
            if len(parts) >= 3:
                dyad.add_context(
                    start=float(parts[0]),
                    end=float(parts[1]),
                    label=parts[2],
                    score=float(parts[3]) if len(parts) > 3 else 0.0,
                )

    dyad.align(target_hz=hz)
    dyad.zscore()

    analyzer = DynamicAnalyzer(
        window_size=args.window_size,
        surrogate_n=args.surrogates,
        max_lag_sec=args.max_lag,
        seed=args.seed,
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(dyad)

    output_path = args.output or "results.json"
    results.export_viewer_json(output_path)
    print(f"  Results exported to: {output_path}")

    # Summary
    print(f"\n  Cascade edges (significant): {len(results.cascade_graph['edges'])}")
    for edge in results.cascade_graph["edges"]:
        print(
            f"    {edge['from']} -> {edge['to']}: "
            f"lag={edge['lag_sec']:.1f}s, "
            f"p={edge['p_value']:.4f}"
        )
    if results.prediction:
        print(
            f"\n  Prediction: delta-AUC = {results.prediction['mean_delta_auc']:.3f}"
        )


def cmd_demo(args: argparse.Namespace) -> None:
    """Run analysis on synthetic data with known ground truth."""
    print("  Generating synthetic dyad (behavior leads neural by 12s, 30% noise)...")
    ds = generate_ground_truth_dyad(
        lead_modality="behavior",
        lag_modality="neural",
        true_lag_sec=12.0,
        noise_ratio=0.3,
        duration_sec=300,
        hz=1.0,
        seed=42,
    )

    ds.align(target_hz=1.0)
    ds.zscore()

    ds.add_context(start=0, end=150, label="PreTask")
    ds.add_context(start=150, end=300, label="Task")

    analyzer = DynamicAnalyzer(
        window_size=10,
        surrogate_n=args.surrogates,
        max_lag_sec=30.0,
        seed=42,
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(ds)

    output_path = args.output or "demo_results.json"
    results.export_viewer_json(output_path)
    print(f"  Results exported to: {output_path}")

    # Ground truth check
    gt = ds._ground_truth
    print(f"\n  Ground truth: {gt['lead']} leads {gt['lag']} by {gt['true_lag_sec']}s")
    print(f"  Cascade edges found: {len(results.cascade_graph['edges'])}")
    for edge in results.cascade_graph["edges"]:
        print(
            f"    {edge['from']} -> {edge['to']}: "
            f"lag={edge['lag_sec']:.1f}s, "
            f"p={edge['p_value']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multisync",
        description="multisync-core: Dynamic process analysis for multimodal synchrony.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze multi-modal dyadic data.")
    p_analyze.add_argument("-i", "--input", required=True, help="Comma-separated CSV paths.")
    p_analyze.add_argument("-n", "--names", help="Comma-separated modality names.")
    p_analyze.add_argument("--hz", default=1.0, help="Target sampling rate.")
    p_analyze.add_argument("-o", "--output", default="results.json", help="Output JSON path.")
    p_analyze.add_argument("--window-size", type=int, default=10, help="WCC window size.")
    p_analyze.add_argument("--surrogates", type=int, default=500, help="Number of surrogates.")
    p_analyze.add_argument("--max-lag", type=float, default=30.0, help="Max CCF lag (sec).")
    p_analyze.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_analyze.add_argument(
        "--contexts", nargs="*", help="Context labels: start,end,label[,score]."
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # demo
    p_demo = sub.add_parser("demo", help="Run analysis on synthetic ground-truth data.")
    p_demo.add_argument("-o", "--output", default="demo_results.json", help="Output path.")
    p_demo.add_argument("--surrogates", type=int, default=500, help="Number of surrogates.")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
