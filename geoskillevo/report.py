"""Generate Markdown tables and standalone SVG figures from results.json."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

METHOD_ORDER = (
    "random", "semantic", "frequency", "logistic", "temperature",
    "isotonic", "fewshot", "isotonic_budget",
)
COLORS = {
    "random": "#9aa0a6",
    "semantic": "#e8a33d",
    "frequency": "#8e7cc3",
    "logistic": "#4a86c5",
    "temperature": "#5aa469",
    "isotonic": "#c0392b",
    "fewshot": "#7f3c8d",
    "isotonic_budget": "#2c3e50",
}


def _svg(width: int, height: int, body: str, background: str = "#ffffff") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">'
        f'<rect width="{width}" height="{height}" fill="{background}"/>'
        f"{body}</svg>"
    )


def _text(x: float, y: float, value: str, size: int = 12, anchor: str = "start",
          weight: str = "normal", fill: str = "#222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#888",
          width: float = 1.0, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def _rect(x: float, y: float, width: float, height: float, fill: str,
          stroke: str = "none", radius: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
    )


def _polyline(points: list[tuple[float, float]], stroke: str, width: float = 2.0) -> str:
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{stroke}" stroke-width="{width}"/>'


def _bar_chart(methods: dict, path: Path) -> None:
    width, height = 1040, 400
    panels = [("success", "Task success rate"), ("score", "Mean task score")]
    body = []
    panel_width = 380
    for panel_index, (key, title) in enumerate(panels):
        x0 = 90 + panel_index * 480
        y0, plot_h = 60, 250
        body.append(_text(x0, 30, title, 15, weight="bold"))
        body.append(_line(x0, y0 + plot_h, x0 + panel_width, y0 + plot_h, "#555", 1.2))
        body.append(_line(x0, y0, x0, y0 + plot_h, "#555", 1.2))
        for tick in range(0, 11, 2):
            value = tick / 10
            y = y0 + plot_h - value * plot_h
            body.append(_line(x0 - 4, y, x0, y, "#555"))
            body.append(_text(x0 - 8, y + 4, f"{value:.1f}", 10, "end", fill="#555"))
            if panel_index == 0:
                body.append(_line(x0, y, x0 + panel_width, y, "#eee", 0.8))
        count = len(METHOD_ORDER)
        slot = panel_width / count
        for index, name in enumerate(METHOD_ORDER):
            row = methods.get(name)
            if not row:
                continue
            value = row[key]
            bar_w = slot * 0.62
            x = x0 + index * slot + (slot - bar_w) / 2
            y = y0 + plot_h - value * plot_h
            body.append(_rect(x, y, bar_w, value * plot_h, COLORS.get(name, "#888")))
            if key == "success" and "success_ci" in row:
                low, high = row["success_ci"]
                cx = x + bar_w / 2
                body.append(_line(cx, y0 + plot_h - low * plot_h,
                                  cx, y0 + plot_h - high * plot_h, "#222", 1.2))
                body.append(_line(cx - 5, y0 + plot_h - low * plot_h,
                                  cx + 5, y0 + plot_h - low * plot_h, "#222", 1.2))
                body.append(_line(cx - 5, y0 + plot_h - high * plot_h,
                                  cx + 5, y0 + plot_h - high * plot_h, "#222", 1.2))
            body.append(_text(x + bar_w / 2, y - 6, f"{value:.2f}", 10, "middle"))
            label = name.replace("_", "\n")
            for line_index, part in enumerate(label.split("\n")):
                body.append(_text(x + bar_w / 2, y0 + plot_h + 18 + line_index * 12,
                                  part, 9, "middle"))
    path.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _reliability_chart(seeds: list[dict], path: Path) -> None:
    width, height = 560, 620
    x0, y0, size = 80, 60, 360
    body = [
        _text(x0, 30, "Reliability of skill-reuse probabilities", 15, weight="bold"),
        _line(x0, y0 + size, x0 + size, y0 + size, "#555", 1.2),
        _line(x0, y0, x0, y0 + size, "#555", 1.2),
        _line(x0, y0 + size, x0 + size, y0, "#bbb", 1.0, "5,5"),
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        x = x0 + value * size
        y = y0 + size - value * size
        body.append(_text(x, y0 + size + 18, f"{value:.1f}", 10, "middle", fill="#555"))
        body.append(_text(x0 - 10, y + 4, f"{value:.1f}", 10, "end", fill="#555"))
    body.append(_text(x0 + size / 2, y0 + size + 42, "Predicted probability", 12, "middle"))
    body.append(
        f'<text x="{x0 - 52}" y="{y0 + size / 2}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 52} {y0 + size / 2})">Observed frequency</text>'
    )
    series = ("semantic", "logistic", "isotonic", "fewshot")
    for series_index, name in enumerate(series):
        bins: dict[int, list[tuple[float, float, float]]] = {}
        for seed in seeds:
            for row in seed["methods"][name]["ranking"].get("reliability", []):
                bins.setdefault(row["bin"], []).append(
                    (row["confidence"], row["accuracy"], row["count"])
                )
        points = []
        for index in sorted(bins):
            rows = bins[index]
            confidence = sum(row[0] for row in rows) / len(rows)
            accuracy = sum(row[1] for row in rows) / len(rows)
            points.append((x0 + confidence * size, y0 + size - accuracy * size))
        body.append(_polyline(points, COLORS.get(name, "#333")))
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{COLORS.get(name)}"/>')
        legend_y = y0 + size + 55 + series_index * 22
        body.append(_line(x0, legend_y - 4, x0 + 26, legend_y - 4, COLORS.get(name), 3))
        body.append(_text(x0 + 34, legend_y, name, 11))
    path.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _ablation_chart(ablation: dict, path: Path) -> None:
    names = [name for name in ("semantic", "structural", "history", "curated", "full") if name in ablation]
    width, height = 760, 360
    plots = [("success", "Top-1 success"), ("auroc", "Ranking AUROC")]
    body = []
    for panel_index, (metric, title) in enumerate(plots):
        x0, y0, plot_w, plot_h = 90 + panel_index * 380, 60, 300, 220
        body.append(_text(x0, 30, title, 14, weight="bold"))
        body.append(_line(x0, y0 + plot_h, x0 + plot_w, y0 + plot_h, "#555"))
        body.append(_line(x0, y0, x0, y0 + plot_h, "#555"))
        slot = plot_w / max(1, len(names))
        for index, name in enumerate(names):
            value = ablation[name][metric]
            bar_w = slot * 0.62
            x = x0 + index * slot + (slot - bar_w) / 2
            color = "#c0392b" if name == "full" else "#4a86c5"
            body.append(_rect(x, y0 + plot_h - value * plot_h, bar_w, value * plot_h, color))
            body.append(_text(x + bar_w / 2, y0 + plot_h - value * plot_h - 5,
                              f"{value:.2f}", 10, "middle"))
            body.append(_text(x + bar_w / 2, y0 + plot_h + 16, name, 10, "middle"))
        for tick in (0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + plot_h - tick * plot_h
            body.append(_line(x0 - 3, y, x0, y, "#555"))
            body.append(_text(x0 - 8, y + 4, f"{tick:.2f}", 9, "end", fill="#555"))
    path.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _memory_chart(stream: dict, path: Path) -> None:
    modes = [name for name in ("none", "flat", "tiered", "tiered_dedup", "full") if name in stream]
    width, height = 920, 360
    panels = [
        ("success", "Task success", lambda row: row["summary"]["success"]),
        ("ece", "Calibration error (ECE)", lambda row: row["ece"]),
        ("chars", "Memory footprint (chars)", lambda row: row["memory"]["characters"] / 60000.0),
    ]
    body = []
    for panel_index, (key, title, getter) in enumerate(panels):
        x0, y0, plot_w, plot_h = 80 + panel_index * 300, 60, 220, 220
        body.append(_text(x0, 30, title, 13, weight="bold"))
        body.append(_line(x0, y0 + plot_h, x0 + plot_w, y0 + plot_h, "#555"))
        body.append(_line(x0, y0, x0, y0 + plot_h, "#555"))
        slot = plot_w / max(1, len(modes))
        for index, mode in enumerate(modes):
            value = getter(stream[mode])
            bar_w = slot * 0.62
            x = x0 + index * slot + (slot - bar_w) / 2
            color = "#7f8c8d" if mode == "none" else ("#c0392b" if mode == "full" else "#4a86c5")
            body.append(_rect(x, y0 + plot_h - value * plot_h, bar_w, value * plot_h, color))
            body.append(_text(x + bar_w / 2, y0 + plot_h - value * plot_h - 5,
                              f"{value:.2f}", 9, "middle"))
            label = "tiered+dedup" if mode == "tiered_dedup" else mode
            body.append(_text(x + bar_w / 2, y0 + plot_h + 16,
                              label, 8, "middle"))
        body.append(_text(x0 - 8, y0 + 4, "1.0", 8, "end", fill="#555"))
        body.append(_text(x0 - 8, y0 + plot_h + 4, "0.0", 8, "end", fill="#555"))
    path.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _heatmap(matrix: dict, path: Path) -> None:
    rows = ["landcover", "crop", "water", "flood", "change", "urban", "burn"]
    rows = [row for row in rows if row in matrix]
    columns = ["landcover", "crop", "water", "flood", "change", "urban", "burn"]
    columns = [column for column in columns if any(column in row for row in matrix.values())]
    cell_w, cell_h = 78, 46
    width = 150 + cell_w * len(columns)
    height = 90 + cell_h * len(rows)
    body = [_text(20, 28, "Probe utility (top) and compatibility (bottom)", 14, weight="bold")]
    for column_index, column in enumerate(columns):
        body.append(_text(150 + column_index * cell_w + cell_w / 2, 62,
                          column, 10, "middle"))
    for row_index, row in enumerate(rows):
        y = 80 + row_index * cell_h
        body.append(_text(140, y + cell_h / 2 + 4, row, 10, "end"))
        for column_index, column in enumerate(columns):
            cell = matrix.get(row, {}).get(column)
            x = 150 + column_index * cell_w
            if not cell:
                body.append(_rect(x, y, cell_w - 2, cell_h - 2, "#f5f5f5"))
                continue
            utility = cell["utility"]
            strength = max(0.0, min(1.0, utility))
            if utility < 0:
                red, green, blue = 220, 230, 245
            else:
                red = int(255 - 80 * strength)
                green = int(255 - 150 * strength)
                blue = int(255 - 190 * strength)
            body.append(_rect(x, y, cell_w - 2, cell_h - 2, f"rgb({red},{green},{blue})"))
            body.append(_text(x + cell_w / 2 - 1, y + 18, f"{utility:+.2f}", 10, "middle"))
            body.append(_text(x + cell_w / 2 - 1, y + 33,
                              f"c={cell['compatibility']:.2f}", 8, "middle", fill="#555"))
    path.write_text(_svg(width, height, "".join(body)), encoding="utf-8")


def _format(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def generate(results_path: str | Path = "experiments/geoskillevo/results.json",
             output: str | Path | None = None) -> Path:
    results_path = Path(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    aggregate = results["aggregate"]
    seeds = results["seeds"]
    output = Path(output) if output else results_path.parent
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _bar_chart(aggregate["methods"], figures / "methods.svg")
    _reliability_chart(seeds, figures / "reliability.svg")
    _ablation_chart(aggregate.get("ablation", {}), figures / "ablation.svg")
    _memory_chart(aggregate["stream"], figures / "memory.svg")
    _heatmap(aggregate.get("transfer_matrix", {}), figures / "transfer_heatmap.svg")

    lines = [
        "# GeoSkillEvo experiment report",
        "",
        f"Seeds: {', '.join(str(seed) for seed in results['config']['seeds'])}. "
        f"Tasks per seed: {seeds[0]['data']['tasks']}. "
        f"Mean skill-library size: {aggregate['library_size']:.1f}. "
        f"Total runtime: {aggregate['elapsed_seconds']:.0f} s.",
        "",
        "## 1. Skill selection quality",
        "",
        "| Method | Success | 95% CI | Score | Negative transfer | AUROC | AUPRC | Brier | NLL | ECE | P@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_ORDER:
        row = aggregate["methods"].get(name)
        if not row:
            continue
        ranking = row["ranking"]
        lines.append(
            f"| {name} | {_format(row['success'])} | "
            f"[{_format(row['success_ci'][0])}, {_format(row['success_ci'][1])}] | "
            f"{_format(row['score'])} | {_format(row['negative_transfer'])} | "
            f"{_format(ranking['auroc'])} | {_format(ranking['auprc'])} | "
            f"{_format(ranking['brier'])} | {_format(ranking['nll'])} | "
            f"{_format(ranking['ece'])} | {_format(ranking['p@1'])} |"
        )
    oracle = aggregate["oracle"]
    lines += [
        f"| oracle (candidate upper bound) | {_format(oracle['success'])} | "
        f"[{_format(oracle['success_ci'][0])}, {_format(oracle['success_ci'][1])}] | "
        f"{_format(oracle['score'])} | 0.000 | - | - | - | - | - | - |",
        "",
        "![Method comparison](figures/methods.svg)",
        "",
        "## 2. Calibration",
        "",
        "Reliability diagram over all target-task candidate skills, averaged across seeds. "
        "The diagonal is perfect calibration.",
        "",
        "![Reliability](figures/reliability.svg)",
        "",
        "## 3. Feature ablation",
        "",
        "All variants use isotonic calibration and the same candidate pool; only the "
        "discriminator features change.",
        "",
        "| Feature set | Success | Score | AUROC | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in aggregate.get("ablation", {}).items():
        lines.append(
            f"| {name} | {_format(row['success'])} | {_format(row['score'])} | "
            f"{_format(row['auroc'])} | {_format(row['ece'])} |"
        )
    lines += [
        "",
        "![Ablation](figures/ablation.svg)",
        "",
        "## 4. Cross-task transfer diagnosis",
        "",
        "| Gate strategy | Success | Score | Negative transfer |",
        "|---|---:|---:|---:|",
    ]
    for name, row in aggregate["selection"].items():
        lines.append(
            f"| {name} | {_format(row['success'])} | {_format(row['score'])} | "
            f"{_format(row['negative_transfer'])} |"
        )
    lines += [
        "",
        "### Source-to-target transferability",
        "",
        "Each cell reports the mean probe utility and structural compatibility estimated "
        "on the target development split.",
        "",
        "![Transfer heatmap](figures/transfer_heatmap.svg)",
        "",
        "### Representative source-to-target pairs",
        "",
        "| Pair | Strategy | Success | Negative transfer | Blocked skills | Library |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair, strategies in aggregate.get("pairs", {}).items():
        for strategy in ("none", "semantic", "proposed", "oracle"):
            row = strategies.get(strategy)
            if not row:
                continue
            lines.append(
                f"| {pair} | {strategy} | {_format(row['success'])} | "
                f"{_format(row['negative_transfer'])} | {row['blocked']:.1f} | "
                f"{row['library']:.1f} |"
            )
    lines += [
        "",
        "## 5. Tiered experience memory",
        "",
        "Prequential evaluation on a held-out stream. Each method selects a skill before "
        "observing the outcome and then ingests the outcome.",
        "",
        "| Memory mode | Success | Score | ECE | Retained records | Characters | Duplicate ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in aggregate["stream"].items():
        memory = row["memory"]
        lines.append(
            f"| {name} | {_format(row['summary']['success'])} | "
            f"{_format(row['summary']['score'])} | {_format(row['ece'])} | "
            f"{memory['retained']} | {memory['characters']} | "
            f"{_format(memory['duplicate_ratio'])} |"
        )
    lines += [
        "",
        "![Memory](figures/memory.svg)",
        "",
        "## 6. Paired tests",
        "",
        "| Comparison | Success delta | McNemar p | Score delta | Permutation p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in aggregate["paired"].items():
        lines.append(
            f"| {name} | {_format(row['success_delta'])} | {_format(row['success_mcnemar_p'], 4)} | "
            f"{_format(row['score_delta'])} | {_format(row['score_permutation_p'], 4)} |"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m geoskillevo.experiment --seeds 0,1,2 --output experiments/geoskillevo",
        "python -m geoskillevo.report",
        "```",
        "",
    ]
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build GeoSkillEvo report and figures")
    parser.add_argument("--results", default="experiments/geoskillevo/results.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = generate(args.results, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
