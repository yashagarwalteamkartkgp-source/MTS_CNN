"""
run_experiment.py — Level 2 Experiment Orchestrator
----------------------------------------------------
Single-command entry point that executes the full evaluation suite:

  Step 1 — Deep Learning Pipeline  (MTS-CNN + diagnostic layer)
  Step 2 — Classical Baselines     (SVM linear, SVM RBF, Random Forest)
  Step 3 — Ablation Experiments    (no-diagnostic, single-channel CNN)
  Step 4 — Full Comparison         (table + bar chart + JSON report)

Usage:
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --config configs/default.yaml --skip-deep
    python run_experiment.py --config configs/default.yaml --steps baselines ablation
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.run_pipeline import run_pipeline
from src.pipeline.run_baselines import run_baselines
from src.pipeline.run_ablation import run_ablation
from src.evaluation.comparison import ModelComparison

# ─────────────────────────────────────────────────────────────────────────────
# Logging — console + persistent file
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('outputs/logs', exist_ok=True)
log_file = f"outputs/logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w'),
    ]
)
logger = logging.getLogger('Orchestrator')

BANNER = """
╔══════════════════════════════════════════════════════════╗
║     MTS-CNN Fault Detection — Full Experiment Suite      ║
╚══════════════════════════════════════════════════════════╝"""

STEP_NAMES = {
    'deep':      'Step 1 — Deep Learning Pipeline (MTS-CNN)',
    'baselines': 'Step 2 — Classical Baselines (SVM + RF)',
    'ablation':  'Step 3 — Ablation Experiments',
    'compare':   'Step 4 — Full Model Comparison',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str):
    width = 62
    logger.info("")
    logger.info("┌" + "─" * width + "┐")
    logger.info("│  " + text.ljust(width - 2) + "│")
    logger.info("└" + "─" * width + "┘")


def run_step(name: str, fn, config_path: str, timings: dict, dry_run: bool = False):
    """Execute one pipeline step with timing, error isolation, and logging."""
    banner(STEP_NAMES[name])
    if dry_run:
        logger.info("[DRY RUN] Skipping execution.")
        return None

    t0 = time.time()
    try:
        result = fn(config_path)
        elapsed = time.time() - t0
        timings[name] = {'status': 'OK', 'elapsed_s': round(elapsed, 2)}
        logger.info(f"✓  {STEP_NAMES[name]} completed in {elapsed:.1f}s")
        return result
    except Exception:
        elapsed = time.time() - t0
        timings[name] = {'status': 'FAILED', 'elapsed_s': round(elapsed, 2)}
        logger.error(f"✗  {STEP_NAMES[name]} FAILED after {elapsed:.1f}s")
        logger.error(traceback.format_exc())
        return None


def merge_results(*result_dicts) -> dict:
    """Merge several metrics dicts, skipping Nones."""
    merged = {}
    for d in result_dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    merged[k] = v
    return merged


def save_final_report(all_results: dict, timings: dict, config_path: str):
    """Write the master JSON report to outputs/."""
    report = {
        'config': config_path,
        'generated_at': datetime.now().isoformat(),
        'timings': timings,
        'results': {
            model: {k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))}
            for model, metrics in all_results.items()
        }
    }
    path = 'outputs/final_report.json'
    os.makedirs('outputs', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Final report saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(config_path: str, steps: list, dry_run: bool = False):
    logger.info(BANNER)
    logger.info(f"Config      : {config_path}")
    logger.info(f"Steps       : {', '.join(steps)}")
    logger.info(f"Log file    : {log_file}")
    logger.info(f"Dry run     : {dry_run}")

    timings = {}
    all_results = {}

    t_total = time.time()

    # ── Step 1 — Deep Learning Pipeline ───────────────────────────────────
    if 'deep' in steps:
        run_step('deep', run_pipeline, config_path, timings, dry_run)
        # run_pipeline saves the trained model; ablation/comparison pick
        # up metrics from the baseline_results / ablation_results JSON files.

    # ── Step 2 — Classical Baselines ──────────────────────────────────────
    if 'baselines' in steps:
        baseline_results = run_step('baselines', run_baselines, config_path, timings, dry_run)
        if baseline_results:
            all_results.update(baseline_results)

    # ── Step 3 — Ablation ─────────────────────────────────────────────────
    if 'ablation' in steps:
        ablation_results = run_step('ablation', run_ablation, config_path, timings, dry_run)
        if ablation_results:
            all_results.update(ablation_results)

    # ── Step 4 — Comparison (always runs if we have any results) ──────────
    if 'compare' in steps:
        banner(STEP_NAMES['compare'])

        # Also pull in any results persisted to disk from earlier runs
        for json_file in ['outputs/baseline_results.json',
                          'outputs/ablation_results.json',
                          'outputs/comparison.json']:
            if os.path.exists(json_file):
                with open(json_file) as f:
                    data = json.load(f)
                for model, metrics in data.items():
                    if isinstance(metrics, dict) and model not in all_results:
                        all_results[model] = metrics

        if all_results:
            comp = ModelComparison(all_results)
            comp.run(plot_dir='outputs/plots',
                     json_path='outputs/comparison.json')
            timings['compare'] = {'status': 'OK'}
        else:
            logger.warning("No results available for comparison yet.")
            timings['compare'] = {'status': 'SKIPPED'}

    # ── Final report ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    report_path = save_final_report(all_results, timings, config_path)

    # ── Timing summary ────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"{'Step':<35} {'Status':>8}  {'Time':>8}")
    logger.info("─" * 60)
    for step, info in timings.items():
        label   = STEP_NAMES.get(step, step)[:34]
        status  = info['status']
        elapsed = f"{info.get('elapsed_s', 0):.1f}s"
        marker  = '✓' if status == 'OK' else ('~' if status == 'SKIPPED' else '✗')
        logger.info(f"{marker}  {label:<33} {status:>8}  {elapsed:>8}")
    logger.info("─" * 60)
    logger.info(f"Total elapsed: {total_elapsed:.1f}s")
    logger.info(f"Final report : {report_path}")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

ALL_STEPS = ['deep', 'baselines', 'ablation', 'compare']

def main():
    parser = argparse.ArgumentParser(
        description='MTS-CNN Fault Detection — Full Experiment Orchestrator'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--steps', nargs='+', choices=ALL_STEPS, default=ALL_STEPS,
        metavar='STEP',
        help=f"Steps to run (default: all). Choices: {ALL_STEPS}"
    )
    parser.add_argument(
        '--skip-deep', action='store_true',
        help='Shorthand to skip the deep learning training step'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print step plan without executing anything'
    )
    args = parser.parse_args()

    steps = args.steps
    if args.skip_deep and 'deep' in steps:
        steps = [s for s in steps if s != 'deep']

    orchestrate(
        config_path=args.config,
        steps=steps,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
