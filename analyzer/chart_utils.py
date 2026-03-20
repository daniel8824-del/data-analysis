"""공용 차트 유틸리티 - Plotly / matplotlib 차트 생성."""
import os
import io
import base64
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils

# ---------------------------------------------------------------------------
# 한글 폰트 설정
# ---------------------------------------------------------------------------
_FONT_PATH = None

def get_korean_font_path():
    global _FONT_PATH
    if _FONT_PATH:
        return _FONT_PATH
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        os.path.join(os.path.dirname(__file__), "..", "fonts", "NanumGothic.ttf"),
        "C:/Windows/Fonts/malgun.ttf",
        "/app/fonts/NanumGothic.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            _FONT_PATH = p
            return _FONT_PATH
    return None


def setup_matplotlib_korean():
    fp = get_korean_font_path()
    if fp:
        try:
            prop = fm.FontProperties(fname=fp)
            name = prop.get_name()
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = name
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False


setup_matplotlib_korean()

# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def plotly_to_json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plotly_donut(labels, values, title="", colors=None, hole=0.5):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=hole,
        textinfo="percent+label",
        marker=dict(colors=colors) if colors else {},
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      margin=dict(t=60, b=30, l=30, r=30),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def plotly_bar_h(labels, values, title="", color=None, text_suffix="건"):
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        text=[f"{v:,}{text_suffix}" for v in values],
        textposition="outside",
        marker_color=color or "#4361ee",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      margin=dict(t=60, b=30, l=120, r=60),
                      xaxis_title="건수", yaxis=dict(autorange="reversed"))
    return fig


def plotly_line(x, y, title="", xlabel="", ylabel=""):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      margin=dict(t=60, b=50, l=60, r=30))
    return fig


def plotly_heatmap(z, x_labels, y_labels, title=""):
    fig = go.Figure(go.Heatmap(z=z, x=x_labels, y=y_labels,
                                colorscale="RdBu_r", texttemplate="%{z:.2f}"))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      margin=dict(t=60, b=50, l=100, r=30))
    return fig


def plotly_histogram(values, title="", xlabel="", nbins=30):
    fig = go.Figure(go.Histogram(x=values, nbinsx=nbins, marker_color="#6366F1"))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      xaxis_title=xlabel, yaxis_title="빈도",
                      margin=dict(t=60, b=50, l=60, r=30))
    return fig


def plotly_network(G, pos=None, title="", node_colors=None, node_sizes=None):
    """NetworkX 그래프 → Plotly Figure."""
    import networkx as nx
    if pos is None:
        pos = nx.spring_layout(G, k=1.5, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.5, color="#888"), hoverinfo="none")
    nodes = list(G.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    sizes = node_sizes or [20] * len(nodes)
    colors = node_colors or ["#4361ee"] * len(nodes)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes, textposition="top center", textfont=dict(size=10),
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      margin=dict(t=60, b=30, l=30, r=30))
    return fig


# ---------------------------------------------------------------------------
# Matplotlib → base64 PNG
# ---------------------------------------------------------------------------

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
