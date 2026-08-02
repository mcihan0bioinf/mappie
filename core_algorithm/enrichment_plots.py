import plotly.graph_objects as go
import math

from .run_enrichment import pretty_ddi_term


def truncate_label(text, max_len=28):
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 2] + ".."


def _term_label(row, max_len=28):
    """Row's display term: pretty DDI name for DDI rows, raw term otherwise."""
    term = row.get("term")
    if row.get("category") == "DDI":
        term = pretty_ddi_term(term)
    return truncate_label(term, max_len)


def build_enrichment_bar(rows, dark_plot=False, title=None):
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No significant term", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(
            paper_bgcolor="#f3f6fb",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Arial", size=14),
            margin=dict(l=50, r=20, t=50, b=70),
            height=360,
        )
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="#1f4e6b", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
        return fig

    counts = {}
    for r in rows:
        cat = r.get("category") or "Other"
        counts[cat] = counts.get(cat, 0) + 1

    ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cats = [truncate_label(c, 20) for c, _ in ordered]
    vals = [v for _, v in ordered]

    paper_bg = "#f3f6fb"
    plot_bg = "#ffffff"
    font_color = "#0f172a"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats,
        y=vals,
        marker_color="#1f4e6b",
    ))

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, family="Arial", size=14),
        margin=dict(l=50, r=20, t=50, b=70),
        height=360,
        showlegend=False,
        title=dict(text=title or "Significant terms", x=0.5, font=dict(size=14)),
        xaxis_title="Category",
        yaxis_title="Number of significant terms",
        yaxis=dict(rangemode="tozero"),
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#1f4e6b", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_enrichment_dotplot(rows, k, category_label=None, top_n=20):
    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No significant term", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(
            paper_bgcolor="#f3f6fb",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Arial", size=13),
            margin=dict(l=80, r=20, t=50, b=40),
            height=360,
        )
        return fig

    use_adj = any(r.get("adj_p") is not None for r in rows)
    top = sorted(rows, key=lambda r: float(r.get("adj_p") or r.get("p_value", "1e9")))[:top_n]
    terms = [_term_label(r) for r in top]
    pvals = []
    sizes = []
    hit_counts = []
    for r in top:
        try:
            p = float(r.get("adj_p") or r.get("p_value"))
        except Exception:
            p = 1.0
        pvals.append(-math.log10(p) if p > 0 else 0.0)
        hits = r.get("observed_hits") or 0
        try:
            hits = float(hits)
        except Exception:
            hits = 0.0
        hit_counts.append(int(hits) if hits > 0 else 0)
        ratio = hits / k if k else hits
        sizes.append(ratio)

    # Clamp sizes to a fixed range (cap max to avoid oversized bubbles).
    size_min, size_max = 10.0, 30.0
    s_min = s_max = None
    def map_ratio_to_size(ratio):
        if s_min is None or s_max is None or s_max == s_min:
            return size_min
        return size_min + (math.sqrt(ratio - s_min) * (size_max - size_min) / math.sqrt(s_max - s_min))

    if sizes:
        s_min, s_max = min(sizes), max(sizes)
        if s_max == s_min:
            sizes = [size_min for _ in sizes]
        else:
            sizes = [map_ratio_to_size(s) for s in sizes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pvals,
        y=terms,
        mode="markers",
        marker=dict(size=sizes, color="#1f4e6b", opacity=0.85),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Size legend: bind to already-rendered marker sizes so legend matches plot 1:1.
    hit_to_sizes = {}
    for h, s in zip(hit_counts, sizes):
        if h <= 0:
            continue
        hit_to_sizes.setdefault(h, []).append(float(s))

    uniq_hits = sorted(hit_to_sizes.keys())
    if len(uniq_hits) >= 2:
        legend_hits = [uniq_hits[0], uniq_hits[-1]]
    else:
        legend_hits = uniq_hits

    # In-plot size key: top-right white box with marker + left-aligned label.
    if legend_hits:
        y_positions = [0.70] if len(legend_hits) == 1 else [0.72, 0.32]
        for i, h in enumerate(legend_hits):
            size = max(hit_to_sizes.get(h, [size_min]))
            fig.add_trace(go.Scatter(
                x=[0.16],
                y=[y_positions[i]],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker=dict(size=size, color="#1f4e6b", opacity=0.85),
                hoverinfo="skip",
                showlegend=False,
                cliponaxis=False,
            ))
            fig.add_annotation(
                x=0.34,
                y=y_positions[i],
                xref="x2",
                yref="y2",
                text=f"{h} neighbours",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
                font=dict(size=13, color="#0f172a"),
            )
    title = f"{category_label} Dotplot" if category_label else "Category Dotplot"
    x_title = "-log10(adj p-value)" if use_adj else "-log10(p-value)"
    fig.update_layout(
        paper_bgcolor="#f3f6fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Arial", size=13),
        margin=dict(l=80, r=20, t=50, b=40),
        height=360,
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis_title=x_title,
        yaxis_title="Top terms",
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(domain=[0.0, 1.0]),
        xaxis2=dict(
            domain=[0.78, 0.98],
            range=[0, 1],
            visible=False,
            fixedrange=True,
        ),
        yaxis2=dict(
            domain=[0.70, 0.98],
            range=[0, 1],
            visible=False,
            fixedrange=True,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", gridwidth=0.5, fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", gridwidth=0.5, fixedrange=True)
    fig.add_shape(
        type="rect",
        xref="x2",
        yref="y2",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#cbd5e1", width=1),
        fillcolor="rgba(255,255,255,0.92)",
        layer="below",
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#1f4e6b", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_enrichment_heatmap(rows, neigh_ppikeys, neigh_labels, top_n=20, p_thresh=None):
    if not rows or not neigh_ppikeys:
        fig = go.Figure()
        fig.add_annotation(text="No significant term", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
        fig.update_layout(
            paper_bgcolor="#f3f6fb",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Arial", size=13),
            margin=dict(l=80, r=20, t=50, b=40),
            height=360,
        )
        return fig

    top = sorted(rows, key=lambda r: float(r.get("adj_p") or r.get("p_value", "1e9")))[:top_n]
    term_labels = [_term_label(r) for r in top]

    col_labels = [truncate_label(x, 18) for x in (neigh_labels or [str(i + 1) for i in range(len(neigh_ppikeys))])]
    ppikeys = neigh_ppikeys

    z = []
    for r in top:
        term_ppis = set(r.get("ppikeys") or [])
        row_vals = [1 if p in term_ppis else 0 for p in ppikeys]
        z.append(row_vals)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=col_labels,
        y=term_labels,
        colorscale=[
            [0.0, "#e5e7eb"],
            [0.5, "#e5e7eb"],
            [0.5, "#1f4e6b"],
            [1.0, "#1f4e6b"],
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        xgap=1,
        ygap=1,
    ))
    fig.update_layout(
        paper_bgcolor="#f3f6fb",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Arial", size=13),
        margin=dict(l=80, r=20, t=50, b=40),
        height=360,
        showlegend=False,
        title=dict(text=f"Top {top_n} terms vs neighbors", x=0.5, font=dict(size=14)),
        xaxis_title="Neighbor PPI",
        yaxis=dict(title="Top terms", title_standoff=20),
        xaxis=dict(tickangle=45),
    )
    # Legend rendered in HTML container (outside plot)
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#1f4e6b", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig
