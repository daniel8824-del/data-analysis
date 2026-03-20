"""EDA (탐색적 데이터 분석) 모듈 - 업로드된 데이터프레임을 자동 분석."""
import os
import logging
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from analyzer.chart_utils import (
    plotly_histogram,
    plotly_bar_h,
    plotly_line,
    plotly_heatmap,
    plotly_to_json,
    fig_to_base64,
    setup_matplotlib_korean,
)

logger = logging.getLogger("review-analyzer.eda")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# 컬럼 타입 탐지
# ---------------------------------------------------------------------------

def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """각 컬럼의 의미적 타입을 탐지한다."""
    types: dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            # 날짜로 파싱 시도
            try:
                pd.to_datetime(df[col], errors="raise")
                types[col] = "datetime"
            except Exception:
                n_unique = df[col].nunique()
                if n_unique <= 20:
                    types[col] = "categorical"
                elif n_unique <= 100:
                    types[col] = "high_cardinality"
                else:
                    types[col] = "text"
    return types


# ---------------------------------------------------------------------------
# Summary HTML 생성
# ---------------------------------------------------------------------------

def _build_summary_html(df: pd.DataFrame, col_types: dict[str, str]) -> str:
    """데이터 개요 + 컬럼 정보 + 기초통계 HTML 생성."""
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())

    # 데이터 개요 테이블
    overview_html = (
        "<h3>📊 데이터 개요</h3>"
        '<table class="table table-bordered table-sm">'
        "<thead><tr><th>항목</th><th>값</th></tr></thead>"
        "<tbody>"
        f"<tr><td>행수</td><td>{n_rows:,}</td></tr>"
        f"<tr><td>열수</td><td>{n_cols:,}</td></tr>"
        f"<tr><td>결측치 총 개수</td><td>{n_missing:,}</td></tr>"
        "</tbody></table>"
    )

    # 컬럼 정보 테이블
    col_rows = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        n_null = int(df[col].isnull().sum())
        n_unique = int(df[col].nunique())
        sample_vals = df[col].dropna().head(3).astype(str).tolist()
        sample_str = ", ".join(sample_vals) if sample_vals else "-"
        # 너무 긴 샘플은 자르기
        if len(sample_str) > 80:
            sample_str = sample_str[:80] + "…"
        detected = col_types.get(col, "-")
        col_rows.append(
            f"<tr><td>{col}</td><td>{dtype_str}</td>"
            f"<td>{detected}</td>"
            f"<td>{n_null:,}</td><td>{n_unique:,}</td>"
            f"<td>{sample_str}</td></tr>"
        )
    col_info_html = (
        "<h3>📋 컬럼 정보</h3>"
        '<div class="table-responsive">'
        '<table class="table table-bordered table-sm">'
        "<thead><tr>"
        "<th>컬럼명</th><th>타입</th><th>탐지 타입</th>"
        "<th>결측치 수</th><th>고유값 수</th><th>샘플값</th>"
        "</tr></thead>"
        "<tbody>" + "".join(col_rows) + "</tbody></table></div>"
    )

    # 기초 통계 (숫자형 컬럼)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc = desc.round(2)
        stat_rows = []
        for idx, row in desc.iterrows():
            vals = "".join(f"<td>{row[c]:,.2f}</td>" for c in desc.columns)
            stat_rows.append(f"<tr><td>{idx}</td>{vals}</tr>")
        headers = "".join(f"<th>{c}</th>" for c in desc.columns)
        stats_html = (
            "<h3>📈 기초 통계량</h3>"
            '<div class="table-responsive">'
            '<table class="table table-bordered table-sm">'
            f"<thead><tr><th>컬럼</th>{headers}</tr></thead>"
            "<tbody>" + "".join(stat_rows) + "</tbody></table></div>"
        )
    else:
        stats_html = "<p>숫자형 컬럼이 없어 기초 통계량을 생성하지 못했습니다.</p>"

    return overview_html + col_info_html + stats_html


# ---------------------------------------------------------------------------
# 차트 생성 헬퍼
# ---------------------------------------------------------------------------

def _select_top_variance_cols(df: pd.DataFrame, cols: list[str], max_n: int) -> list[str]:
    """분산이 큰 순서대로 최대 max_n개 컬럼 선택."""
    if len(cols) <= max_n:
        return cols
    variances = {}
    for col in cols:
        try:
            variances[col] = df[col].dropna().var()
        except Exception:
            variances[col] = 0
    sorted_cols = sorted(variances, key=variances.get, reverse=True)
    return sorted_cols[:max_n]


def _make_chart(fig_or_data, chart_mode: str) -> str:
    """chart_mode 에 따라 plotly JSON 또는 base64 PNG 반환."""
    if chart_mode == "plotly":
        return plotly_to_json(fig_or_data)
    else:
        return fig_to_base64(fig_or_data)


def _generate_histograms(
    df: pd.DataFrame, numeric_cols: list[str], chart_mode: str
) -> list[dict]:
    """숫자형 컬럼 히스토그램 (최대 3개)."""
    selected = _select_top_variance_cols(df, numeric_cols, 3)
    charts = []
    for col in selected:
        try:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            title = f"{col} 분포"
            if chart_mode == "plotly":
                fig = plotly_histogram(series.tolist(), title=title, xlabel=col)
                charts.append({
                    "title": title,
                    "type": "plotly",
                    "data": plotly_to_json(fig),
                })
            else:
                setup_matplotlib_korean()
                mpl_fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(series, bins=30, color="#6366F1", edgecolor="white")
                ax.set_title(title)
                ax.set_xlabel(col)
                ax.set_ylabel("빈도")
                charts.append({
                    "title": title,
                    "type": "image",
                    "data": fig_to_base64(mpl_fig),
                })
        except Exception:
            logger.warning("히스토그램 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_bar_charts(
    df: pd.DataFrame, cat_cols: list[str], chart_mode: str
) -> list[dict]:
    """범주형 컬럼 수평 바 차트 (최대 3개)."""
    # 고유값 수가 많은 순으로 관심도 높은 컬럼 선택
    if len(cat_cols) > 3:
        scored = sorted(cat_cols, key=lambda c: df[c].nunique(), reverse=True)
        cat_cols = scored[:3]
    charts = []
    for col in cat_cols:
        try:
            vc = df[col].value_counts().head(15)
            if len(vc) == 0:
                continue
            labels = vc.index.astype(str).tolist()
            values = vc.values.tolist()
            title = f"{col} 빈도"
            if chart_mode == "plotly":
                fig = plotly_bar_h(labels, values, title=title)
                charts.append({
                    "title": title,
                    "type": "plotly",
                    "data": plotly_to_json(fig),
                })
            else:
                setup_matplotlib_korean()
                mpl_fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.35)))
                ax.barh(labels[::-1], values[::-1], color="#4361ee")
                ax.set_title(title)
                ax.set_xlabel("건수")
                charts.append({
                    "title": title,
                    "type": "image",
                    "data": fig_to_base64(mpl_fig),
                })
        except Exception:
            logger.warning("바 차트 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_line_charts(
    df: pd.DataFrame, dt_cols: list[str], chart_mode: str
) -> list[dict]:
    """날짜 컬럼별 시간 추이 라인 차트."""
    charts = []
    for col in dt_cols:
        try:
            dt_series = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dt_series) == 0:
                continue
            counts = dt_series.dt.date.value_counts().sort_index()
            x = [str(d) for d in counts.index]
            y = counts.values.tolist()
            title = "시간별 데이터 추이"
            if chart_mode == "plotly":
                fig = plotly_line(x, y, title=title, xlabel="날짜", ylabel="건수")
                charts.append({
                    "title": title,
                    "type": "plotly",
                    "data": plotly_to_json(fig),
                })
            else:
                setup_matplotlib_korean()
                mpl_fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, y, marker="o", markersize=3, linewidth=1)
                ax.set_title(title)
                ax.set_xlabel("날짜")
                ax.set_ylabel("건수")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                charts.append({
                    "title": title,
                    "type": "image",
                    "data": fig_to_base64(mpl_fig),
                })
        except Exception:
            logger.warning("라인 차트 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_heatmap(
    df: pd.DataFrame, numeric_cols: list[str], chart_mode: str
) -> list[dict]:
    """상관관계 히트맵 (숫자 컬럼 2개 이상일 때)."""
    if len(numeric_cols) < 2:
        return []
    charts = []
    try:
        corr = df[numeric_cols].corr()
        title = "상관관계 히트맵"
        if chart_mode == "plotly":
            fig = plotly_heatmap(
                z=corr.values.tolist(),
                x_labels=corr.columns.tolist(),
                y_labels=corr.index.tolist(),
                title=title,
            )
            charts.append({
                "title": title,
                "type": "plotly",
                "data": plotly_to_json(fig),
            })
        else:
            setup_matplotlib_korean()
            mpl_fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) * 0.8)))
            im = ax.imshow(corr.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax.set_yticklabels(numeric_cols)
            # 셀에 값 표시
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            charts.append({
                "title": title,
                "type": "image",
                "data": fig_to_base64(mpl_fig),
            })
    except Exception:
        logger.warning("히트맵 생성 실패\n%s", traceback.format_exc())
    return charts


def _detect_outliers(df: pd.DataFrame, col: str) -> tuple[pd.Series, float, float]:
    """IQR 기반 이상치 탐지. (mask, lower, upper) 반환."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    return mask, lower, upper


def _generate_box_plots(
    df: pd.DataFrame, numeric_cols: list[str], chart_mode: str
) -> tuple[list[dict], str]:
    """이상치 탐지 박스 플롯 (최대 3개) + 이상치 요약 텍스트."""
    # 이상치가 많은 컬럼 우선 선택
    outlier_info: list[tuple[str, int, float, float]] = []
    for col in numeric_cols:
        try:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            mask, lower, upper = _detect_outliers(df, col)
            n_outliers = int(mask.sum())
            outlier_info.append((col, n_outliers, lower, upper))
        except Exception:
            pass

    # 이상치가 있는 컬럼만, 이상치 수 기준 정렬
    outlier_info.sort(key=lambda x: x[1], reverse=True)
    selected = [info for info in outlier_info if info[1] > 0][:3]

    charts = []
    details_lines = []
    for col, n_out, lower, upper in selected:
        try:
            title = f"{col} 이상치 탐지"
            series = df[col].dropna()
            pct = n_out / len(df) * 100 if len(df) > 0 else 0
            details_lines.append(
                f"• <b>{col}</b>: 이상치 {n_out:,}건 ({pct:.1f}%) "
                f"[정상 범위: {lower:,.2f} ~ {upper:,.2f}]"
            )
            if chart_mode == "plotly":
                fig = go.Figure(go.Box(y=series.tolist(), name=col,
                                       marker_color="#6366F1",
                                       boxmean="sd"))
                fig.update_layout(
                    title=dict(text=title, font=dict(size=16)),
                    yaxis_title=col,
                    margin=dict(t=60, b=30, l=60, r=30),
                )
                charts.append({
                    "title": title,
                    "type": "plotly",
                    "data": plotly_to_json(fig),
                })
            else:
                setup_matplotlib_korean()
                mpl_fig, ax = plt.subplots(figsize=(5, 4))
                ax.boxplot(series, vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#6366F1", alpha=0.6))
                ax.set_title(title)
                ax.set_ylabel(col)
                charts.append({
                    "title": title,
                    "type": "image",
                    "data": fig_to_base64(mpl_fig),
                })
        except Exception:
            logger.warning("박스 플롯 생성 실패: %s\n%s", col, traceback.format_exc())

    # 이상치가 없는 경우
    if not details_lines:
        # 전체 숫자 컬럼에 대해서도 이상치 없음 표시
        for col, n_out, lower, upper in outlier_info:
            details_lines.append(
                f"• <b>{col}</b>: 이상치 없음 "
                f"[정상 범위: {lower:,.2f} ~ {upper:,.2f}]"
            )

    details_html = (
        "<h3>🔍 이상치 탐지 결과 (IQR 기반)</h3>"
        + ("<br>".join(details_lines) if details_lines else "<p>숫자형 컬럼이 없어 이상치 분석을 수행하지 못했습니다.</p>")
    )
    return charts, details_html


# ---------------------------------------------------------------------------
# 메인 EDA 실행
# ---------------------------------------------------------------------------

def run_eda(
    df: pd.DataFrame,
    job_id: str,
    chart_mode: str = "plotly",
) -> dict:
    """
    EDA 실행 → dict 반환.

    Returns
    -------
    dict with keys: summary_html, charts, details_html, downloads
    """
    # 결과 저장 디렉토리
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ── 1. 컬럼 타입 탐지 ──
    col_types = detect_column_types(df)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    dt_cols = [c for c, t in col_types.items() if t == "datetime"]

    # ── 2. Summary HTML ──
    summary_html = _build_summary_html(df, col_types)

    # ── 3. 차트 생성 ──
    charts: list[dict] = []

    # 히스토그램
    try:
        charts.extend(_generate_histograms(df, numeric_cols, chart_mode))
    except Exception:
        logger.warning("히스토그램 생성 중 오류\n%s", traceback.format_exc())

    # 범주형 바 차트
    try:
        charts.extend(_generate_bar_charts(df, cat_cols, chart_mode))
    except Exception:
        logger.warning("바 차트 생성 중 오류\n%s", traceback.format_exc())

    # 시간 추이
    try:
        charts.extend(_generate_line_charts(df, dt_cols, chart_mode))
    except Exception:
        logger.warning("라인 차트 생성 중 오류\n%s", traceback.format_exc())

    # 상관관계 히트맵
    try:
        charts.extend(_generate_heatmap(df, numeric_cols, chart_mode))
    except Exception:
        logger.warning("히트맵 생성 중 오류\n%s", traceback.format_exc())

    # 박스 플롯 + 이상치 요약
    try:
        box_charts, details_html = _generate_box_plots(df, numeric_cols, chart_mode)
        charts.extend(box_charts)
    except Exception:
        logger.warning("박스 플롯 생성 중 오류\n%s", traceback.format_exc())
        details_html = "<p>이상치 분석 중 오류가 발생했습니다.</p>"

    # ── 4. 전처리 데이터 CSV 저장 ──
    csv_filename = "EDA_결과.csv"
    csv_path = os.path.join(job_dir, csv_filename)
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception:
        logger.warning("CSV 저장 실패\n%s", traceback.format_exc())

    downloads = [{"filename": csv_filename, "label": "전처리 데이터 CSV"}]

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
