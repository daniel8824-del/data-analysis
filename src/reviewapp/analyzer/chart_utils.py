"""공용 차트 유틸리티 - Plotly / matplotlib 차트 생성."""
import os
import io
import base64
import json
import zipfile
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
_FONT_FAMILY = "Pretendard, Noto Sans KR, -apple-system, sans-serif"

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
# Plotly 공통 스타일
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    font=dict(family=_FONT_FAMILY, size=13, color="#333"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=50, b=40, l=20, r=20),
)

# 전문적 컬러 팔레트 (사업계획서 스타일)
CHART_COLORS = [
    "#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6",
]


def _apply_style(fig, **overrides):
    """공통 Plotly 스타일 적용."""
    layout = {**_PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def plotly_to_json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plotly_donut(labels, values, title="", colors=None, hole=0.45):
    total = sum(values) if values else 1
    palette = colors or CHART_COLORS
    custom_text = []
    for lab, val in zip(labels, values):
        pct = val / total * 100
        if pct >= 5:
            custom_text.append(f"{lab}<br>{pct:.1f}%")
        else:
            custom_text.append("")
    hover_text = [f"{lab}: {val:,} ({val/total*100:.1f}%)" for lab, val in zip(labels, values)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=hole,
        text=custom_text,
        textinfo="text",
        textposition="inside",
        textfont=dict(size=12, color="white"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(colors=palette, line=dict(color="white", width=2)),
        sort=False,
    ))
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333"), x=0.5, xanchor="center"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(size=11)),
        height=400,
    )
    return fig


def plotly_bar_h(labels, values, title="", color=None, text_suffix="건"):
    palette = color or CHART_COLORS
    max_val = max(values) if values else 1
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        text=[f"{v:,}{text_suffix}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#555"),
        marker=dict(color=palette, line=dict(color="white", width=1)),
    ))
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333"), x=0.5, xanchor="center"),
        xaxis=dict(title="", range=[0, max_val * 1.22],
                   showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        margin=dict(t=50, b=30, l=90, r=60),
        height=max(300, len(labels) * 42 + 80),
    )
    return fig


def plotly_line(x, y, title="", xlabel="", ylabel=""):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers",
                                line=dict(color="#3498DB", width=2),
                                marker=dict(size=6)))
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333")),
        xaxis_title=xlabel, yaxis_title=ylabel,
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def plotly_heatmap(z, x_labels, y_labels, title=""):
    fig = go.Figure(go.Heatmap(z=z, x=x_labels, y=y_labels,
                                colorscale="RdBu_r", texttemplate="%{z:.2f}"))
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333")),
        margin=dict(t=60, b=50, l=100, r=30),
    )
    return fig


def plotly_histogram(values, title="", xlabel="", nbins=30):
    fig = go.Figure(go.Histogram(x=values, nbinsx=nbins,
                                  marker_color="#3498DB",
                                  marker_line=dict(color="white", width=1)))
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333")),
        xaxis_title=xlabel, yaxis_title="빈도",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def plotly_network(G, pos=None, title="", node_colors=None, node_sizes=None):
    """NetworkX 그래프 -> Plotly Figure."""
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
                            line=dict(width=0.8, color="#ccc"), hoverinfo="none")
    nodes = list(G.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    sizes = node_sizes or [20] * len(nodes)
    colors = node_colors or ["#3498DB"] * len(nodes)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes, textposition="top center", textfont=dict(size=10),
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="white")),
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    _apply_style(fig,
        title=dict(text=title, font=dict(size=14, color="#333")),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=60, b=30, l=30, r=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib -> base64 PNG
# ---------------------------------------------------------------------------

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plotly_save_png(fig, path, width=1000, height=500, scale=2):
    """Plotly figure를 PNG 파일로 저장 (kaleido 사용)."""
    try:
        fig.write_image(path, format="png", width=width, height=height, scale=scale)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ZIP 다운로드 유틸
# ---------------------------------------------------------------------------

def bundle_zip(result_dir, zip_name="분석결과.zip", pattern=None):
    """result_dir 내 파일들을 ZIP으로 묶어 저장. pattern이 있으면 해당 확장자만."""
    import fnmatch
    zip_path = os.path.join(result_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(result_dir):
            if fname.endswith(".zip"):
                continue
            if pattern and not fnmatch.fnmatch(fname, pattern):
                continue
            fpath = os.path.join(result_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    return zip_name
