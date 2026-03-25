"""Generate analysis charts from autosteer pipeline data."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from run import BEHAVIORS_DIR, SteeringVector, _load_vector, _model_slug

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Shared layout defaults for all charts
_LAYOUT_DEFAULTS = dict(
    template="seaborn",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(showgrid=False, linecolor="black", linewidth=1, mirror=False),
    yaxis=dict(showgrid=False, linecolor="black", linewidth=1, mirror=False),
    plot_bgcolor="white",
)


# ── Data Loading ─────────────────────────────────────────────────────────────


def detect_method(behavior_dir: Path, model_slug: str) -> str | None:
    """Auto-detect 'mean' vs 'pca' by checking which layer0 vector file exists."""
    for method in ("mean", "pca"):
        if (behavior_dir / "vectors" / f"{model_slug}_layer0_{method}.pt").exists():
            return method
    return None


def load_all_vectors(behavior_dir: Path, model_slug: str, method: str) -> list[SteeringVector]:
    """Load all layer steering vectors for a behavior, sorted by layer."""
    vectors_dir = behavior_dir / "vectors"
    vectors: list[SteeringVector] = []
    for layer in range(128):  # generous upper bound
        path = vectors_dir / f"{model_slug}_layer{layer}_{method}.pt"
        v = _load_vector(path)
        if v is None:
            if layer > 0:
                break  # past the last layer
            continue
        vectors.append(v)
    return sorted(vectors, key=lambda v: v.layer)





# ── Chart Functions ──────────────────────────────────────────────────────────


def chart_layer_metrics(vectors: list[SteeringVector], behavior: str) -> go.Figure:
    """Plot of normalized magnitude, PC1 explained variance, and probe accuracy per layer."""
    layers = [v.layer for v in vectors]
    magnitudes = [v.magnitude for v in vectors]
    max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0
    mag_normalized = [m / max_mag for m in magnitudes]
    pc1_variance = [v.explained_variances[0] if v.explained_variances else 0.0 for v in vectors]
    probe_acc = [v.probe_accuracy for v in vectors]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=mag_normalized, name="Magnitude (normalized)", mode="lines+markers", marker=dict(size=4, color="#1f77b4"), line=dict(color="#1f77b4")))
    if not all(v == 0.0 for v in pc1_variance):
        fig.add_trace(go.Scatter(x=layers, y=pc1_variance, name="PC1 Explained Variance", mode="lines+markers", marker=dict(size=4, color="#5e2d91"), line=dict(color="#5e2d91")))
    fig.add_trace(go.Scatter(x=layers, y=probe_acc, name="Probe Accuracy", mode="lines+markers", marker=dict(size=4, color="#d62728"), line=dict(color="#d62728")))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"Layer Metrics: {behavior}",
        xaxis_title="Layer",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
    )
    return fig


def chart_stability_consistency(vectors: list[SteeringVector], behavior: str) -> go.Figure | None:
    """Plot per-layer stability, consistency, norm sensitivity, and SNR."""
    stabilities = [v.stability for v in vectors]
    consistencies = [v.consistency for v in vectors]
    snrs = [v.snr for v in vectors]
    norm_sens = [v.norm_sensitivity for v in vectors]
    has_stability = not all(s == 0.0 for s in stabilities)
    has_consistency = not all(c == 0.0 for c in consistencies)
    has_snr = not all(s == 0.0 for s in snrs)
    has_norm_sens = not all(s == 0.0 for s in norm_sens)
    if not has_stability and not has_consistency and not has_snr and not has_norm_sens:
        log.warning("No stability/consistency/SNR/norm-sensitivity data for %s — skipping", behavior)
        return None

    layers = [v.layer for v in vectors]
    fig = go.Figure()
    if has_stability:
        fig.add_trace(go.Scatter(x=layers, y=stabilities, name="Stability", mode="lines+markers", marker=dict(size=4, color="#9467bd"), line=dict(color="#9467bd")))
    if has_consistency:
        fig.add_trace(go.Scatter(x=layers, y=consistencies, name="Consistency", mode="lines+markers", marker=dict(size=4, color="#ff7f0e"), line=dict(color="#ff7f0e")))
    if has_norm_sens:
        fig.add_trace(go.Scatter(x=layers, y=norm_sens, name="Norm Sensitivity", mode="lines+markers", marker=dict(size=4, color="#d62728"), line=dict(color="#d62728")))
    if has_snr:
        fig.add_trace(go.Scatter(x=layers, y=snrs, name="SNR", mode="lines+markers", marker=dict(size=4, color="#2ca02c"), line=dict(color="#2ca02c"), yaxis="y2"))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"Vector Quality: {behavior}",
        xaxis_title="Layer",
        yaxis_title="Cosine Similarity",
        yaxis_range=[0, 1.05],
        yaxis2=dict(title="SNR", overlaying="y", side="right", showgrid=False, linecolor="black", linewidth=1, rangemode="tozero"),
    )
    return fig



def chart_pca_variance_stacked_bar(vectors: list[SteeringVector], behavior: str) -> go.Figure | None:
    """Horizontal stacked bar of PCA explained variance per layer."""
    if not any(v.explained_variances for v in vectors):
        log.warning("No PCA variance data for %s — skipping", behavior)
        return None

    n_components = max(len(v.explained_variances) for v in vectors)
    layer_labels = [str(v.layer) for v in vectors]

    component_colors = [
        "#F28E8E", "#F2C46D", "#A8D8A8", "#8EC5E8", "#C5A3D9",
        "#F2A8C7", "#8ED8D0", "#D4C98E", "#B0A8D8", "#D9AFA0",
    ]

    fig = go.Figure()
    for i in range(n_components):
        values = [
            v.explained_variances[i] * 100 if i < len(v.explained_variances) else 0.0
            for v in vectors
        ]
        color = component_colors[i] if i < len(component_colors) else component_colors[-1]
        fig.add_trace(go.Bar(
            y=layer_labels,
            x=values,
            name=f"PC{i+1}",
            orientation="h",
            marker_color=color,
            text=[f"{val:.0f}%" if val >= 3 else "" for val in values],
            textposition="inside",
            textfont=dict(color="#333333", size=10),
        ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        barmode="stack",
        title=f"PCA Explained Variance by Layer: {behavior}",
        height=max(500, len(vectors) * 22 + 100),
        legend=dict(title="Component", traceorder="normal"),
    )
    fig.update_xaxes(title="Explained Variance (%)", range=[0, 105])
    fig.update_yaxes(title="Layer", autorange="reversed")
    return fig


def chart_adjacent_similarity(all_vectors: dict[str, list[SteeringVector]]) -> go.Figure | None:
    """Plot cosine similarity between adjacent-layer steering vectors, one line per behavior."""
    _COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = go.Figure()
    trace_count = 0
    for idx, (name, vectors) in enumerate(sorted(all_vectors.items())):
        if len(vectors) < 2:
            continue
        sorted_vecs = sorted(vectors, key=lambda v: v.layer)
        transitions: list[str] = []
        similarities: list[float] = []
        for i in range(len(sorted_vecs) - 1):
            a = sorted_vecs[i].vector.float()
            b = sorted_vecs[i + 1].vector.float()
            cos = torch.dot(a / a.norm(), b / b.norm()).item()
            transitions.append(f"{sorted_vecs[i].layer}\u2192{sorted_vecs[i + 1].layer}")
            similarities.append(cos)
        color = _COLORS[idx % len(_COLORS)]
        fig.add_trace(go.Scatter(
            x=transitions,
            y=similarities,
            name=name,
            mode="lines+markers",
            marker=dict(size=4, color=color),
            line=dict(color=color),
        ))
        trace_count += 1

    if trace_count == 0:
        return None
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Adjacent-Layer Vector Similarity",
        xaxis_title="Layer transition (i \u2192 i+1)",
        yaxis_title="Cosine similarity",
        yaxis_range=[-0.1, 1.05],
    )
    return fig


def chart_eval_scores(scored_path: str | Path) -> go.Figure | None:
    """Side-by-side heatmaps of coherence and behavior scores from a scored run file."""
    scored_path = Path(scored_path)
    if not scored_path.exists():
        log.warning("Scored file not found: %s", scored_path)
        return None

    data = json.loads(scored_path.read_text())
    steered = data["steered"]
    if len(steered) < 2:
        log.warning("Insufficient data for eval heatmap — skipping")
        return None

    all_layers = sorted({e["layer"] for e in steered})
    all_alphas = sorted({e["alpha"] for e in steered})
    if len(all_layers) < 2 or len(all_alphas) < 2:
        log.warning("Need at least 2 layers and 2 alphas for eval heatmap — skipping")
        return None

    # Detect the behavior score key (everything except prompt, response, coherence)
    sample_keys = set(steered[0]["results"][0].keys())
    behavior_key = (sample_keys - {"prompt", "response", "coherence"}).pop()

    layer_idx = {l: i for i, l in enumerate(all_layers)}
    alpha_idx = {a: i for i, a in enumerate(all_alphas)}

    coherence_grid = np.full((len(all_layers), len(all_alphas)), np.nan)
    behavior_grid = np.full((len(all_layers), len(all_alphas)), np.nan)

    for entry in steered:
        li = layer_idx[entry["layer"]]
        ai = alpha_idx[entry["alpha"]]
        scores = entry["results"]
        coherence_grid[li, ai] = np.mean([r["coherence"] for r in scores])
        behavior_grid[li, ai] = np.mean([r[behavior_key] for r in scores])

    x_labels = [f"{a:.2f}" for a in all_alphas]
    y_labels = [str(l) for l in all_layers]
    behavior_label = behavior_key.replace("_", " ").title()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Coherence (1=nonsense, 5=coherent)",
            f"{behavior_label} (1=low, 5=high)",
        ),
        horizontal_spacing=0.18,
    )

    for col, (grid, colorscale, title) in enumerate(
        [
            (coherence_grid, [[0, "#f7fbff"], [1, "#08306b"]], "Coherence"),
            (behavior_grid, [[0, "#fff5eb"], [1, "#7f2704"]], behavior_label),
        ],
        start=1,
    ):
        text = [
            [f"{grid[i, j]:.1f}" if not np.isnan(grid[i, j]) else "" for j in range(len(all_alphas))]
            for i in range(len(all_layers))
        ]
        fig.add_trace(
            go.Heatmap(
                z=grid.tolist(),
                x=x_labels,
                y=y_labels,
                text=text,
                texttemplate="%{text}",
                colorscale=colorscale,
                zmin=1,
                zmax=5,
                colorbar=dict(title=title, x=0.42 if col == 1 else 1.02),
                showscale=True,
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="Alpha", row=1, col=col)
        fig.update_yaxes(title_text="Layer", row=1, col=col)

    fig.update_layout(
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")},
        title=f"Eval Scores: {scored_path.stem}",
        height=max(600, len(all_layers) * 25 + 200),
        width=1400,
    )
    return fig


# ── I/O ──────────────────────────────────────────────────────────────────────


def save_figure(fig: go.Figure, path: Path) -> None:
    """Save a Plotly figure as SVG, respecting the figure's own height if set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = path.with_suffix(".svg")
    height = fig.layout.height or 450
    fig.write_image(str(svg_path), format="svg", width=900, height=height)
    log.info("Saved %s", svg_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def analyze(
    model_name: str = "Qwen/Qwen3.5-9B",
    output_dir: str = "charts",
) -> None:
    """Generate all analysis charts for all behaviors."""
    slug = _model_slug(model_name)
    out = Path(output_dir)

    if not BEHAVIORS_DIR.exists():
        log.error("No behaviors/ directory found")
        return
    behavior_dirs = sorted(p for p in BEHAVIORS_DIR.iterdir() if p.is_dir())
    if not behavior_dirs:
        log.error("No behavior directories found")
        return

    all_vectors: dict[str, list[SteeringVector]] = {}

    for bdir in behavior_dirs:
        name = bdir.name
        method = detect_method(bdir, slug)
        if method is None:
            log.warning("No vectors found for %s — skipping", name)
            continue

        vectors = load_all_vectors(bdir, slug, method)
        if len(vectors) < 2:
            log.warning("Too few vectors for %s — skipping", name)
            continue

        all_vectors[name] = vectors
        log.info("Loaded %d vectors for %s (method=%s)", len(vectors), name, method)

        bdir_out = out / name

        # Chart 1: Layer Metrics (PC1 only)
        fig = chart_layer_metrics(vectors, name)
        save_figure(fig, bdir_out / "1_layer_metrics")

        # Chart 2: PCA Variance Stacked Bar
        fig = chart_pca_variance_stacked_bar(vectors, name)
        if fig is not None:
            save_figure(fig, bdir_out / "2_pca_variance")

        # Chart 3: Stability & Consistency
        fig = chart_stability_consistency(vectors, name)
        if fig is not None:
            save_figure(fig, bdir_out / "3_stability_consistency")

        # Chart 4: Eval Scores (if scored file exists)
        scored_path = bdir / "runs" / f"{slug}_{method}_scored.json"
        fig = chart_eval_scores(scored_path)
        if fig is not None:
            save_figure(fig, bdir_out / "4_eval_scores")

    # Chart 5: Adjacent-Layer Vector Similarity (all behaviors on one chart)
    fig = chart_adjacent_similarity(all_vectors)
    if fig is not None:
        save_figure(fig, out / "5_adjacent_similarity")

    log.info("Done. Charts saved to %s/", out)


def main() -> None:
    """Entry point."""
    fire.Fire(analyze)


if __name__ == "__main__":
    main()
