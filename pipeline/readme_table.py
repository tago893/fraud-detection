"""Render the README results tables from artifacts/metrics.json.

python -m pipeline.readme_table        # rewrites the block between the markers in README.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.config import METRICS_PATH, ROOT

README_PATH = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"

_ROWS = (
    ("PR-AUC", "pr_auc", "{:.4f}"),
    ("ROC-AUC", "roc_auc", "{:.4f}"),
    ("Precision", "precision", "{:.3f}"),
    ("Recall", "recall", "{:.3f}"),
    ("F1", "f1", "{:.3f}"),
    ("Cost at t*", "cost", "${:,.2f}"),
    ("Cost at threshold 0.5", "cost_at_0.5", "${:,.2f}"),
    ("Cost if nothing is flagged", "cost_flag_nothing", "${:,.2f}"),
)


def _table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = [header, align, *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def render_results(metrics: dict) -> str:
    """Markdown for the candidate comparison and the val/test table at the chosen threshold."""
    selected = metrics["selected"]
    candidates = [
        [
            f"`{name}`" + (" (selected)" if name == selected else ""),
            f"{scores['val_pr_auc']:.4f}",
            f"{scores['val_roc_auc']:.4f}",
        ]
        for name, scores in metrics["candidates"].items()
    ]
    val, test = metrics["val"], metrics["test"]
    final = [
        [label, fmt.format(val[key]), fmt.format(test[key])]
        for label, key, fmt in _ROWS
    ]
    for part, label in (
        ("tp", "True positives"),
        ("fp", "False positives"),
        ("fn", "False negatives"),
    ):
        final.append(
            [
                label,
                str(val["confusion_matrix"][part]),
                str(test["confusion_matrix"][part]),
            ]
        )
    return (
        f"**Model selection** (validation slice, time-ordered split):\n\n"
        f"{_table(['Candidate', 'Val PR-AUC', 'Val ROC-AUC'], candidates)}\n\n"
        f"**Final metrics** at the validation-chosen threshold t* = {test['threshold']:.2f} "
        f"(test slice scored once; model `{selected}`, version `{metrics['model_version']}`):\n\n"
        f"{_table(['Metric', 'Validation', 'Test'], final)}"
    )


def splice(readme: str, block: str) -> str:
    """Replace the text between the START and END markers; fail loudly if they are missing."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError(f"README is missing the {START} ... {END} markers")
    return pattern.sub(lambda _: f"{START}\n{block}\n{END}", readme)


def update_readme(
    readme_path: Path = README_PATH, metrics_path: Path = METRICS_PATH
) -> None:
    metrics = json.loads(metrics_path.read_text())
    readme_path.write_text(
        splice(readme_path.read_text(encoding="utf-8"), render_results(metrics)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    update_readme()
    print(f"updated {README_PATH}")
