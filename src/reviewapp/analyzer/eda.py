"""EDA (탐색적 데이터 분석) 모듈 - 업로드된 데이터프레임을 자동 분석."""
import os
import json
import logging
import traceback

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from reviewapp.analyzer.chart_utils import (
    plotly_histogram,
    plotly_bar_h,
    plotly_line,
    plotly_heatmap,
    plotly_to_json,
    plotly_save_png,
    bundle_zip,
    _apply_style,
    CHART_COLORS,
)

logger = logging.getLogger("review-analyzer.eda")

WORK_DIR = os.path.join(os.path.expanduser("~"), ".review-analyzer")
RESULT_DIR = os.path.join(WORK_DIR, "results")

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
    n_dup = int(df.duplicated().sum())
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    dt_cols = [c for c, t in col_types.items() if t == "datetime"]
    text_cols = [c for c, t in col_types.items() if t in ("text", "high_cardinality")]

    # 데이터 구조 요약
    overview_rows = (
        f"<tr><td>전체 행 수</td><td >{n_rows:,}</td></tr>"
        f"<tr><td>전체 열 수</td><td >{n_cols:,}</td></tr>"
        f"<tr><td>결측치</td><td >{n_missing:,}건"
        f" ({n_missing/(n_rows*n_cols)*100:.1f}%)</td></tr>"
        f"<tr><td>중복 행</td><td >{n_dup:,}건</td></tr>"
        f"<tr><td>숫자형 컬럼</td><td >{len(numeric_cols)}개</td></tr>"
        f"<tr><td>범주형 컬럼</td><td >{len(cat_cols)}개</td></tr>"
        f"<tr><td>날짜형 컬럼</td><td >{len(dt_cols)}개</td></tr>"
        f"<tr><td>텍스트 컬럼</td><td >{len(text_cols)}개</td></tr>"
    )
    overview_html = (
        '<table>'
        '<thead><tr><th>데이터 구조</th><th>값</th></tr></thead>'
        f'<tbody>{overview_rows}</tbody></table>'
    )

    # 컬럼별 상세
    col_rows = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        n_null = int(df[col].isnull().sum())
        null_pct = n_null / n_rows * 100 if n_rows > 0 else 0
        n_unique = int(df[col].nunique())
        detected = col_types.get(col, "-")
        null_display = f"{n_null:,} ({null_pct:.1f}%)" if n_null > 0 else "0"
        col_rows.append(
            f"<tr><td>{col}</td><td>{dtype_str}</td>"
            f"<td>{detected}</td>"
            f"<td >{null_display}</td>"
            f"<td >{n_unique:,}</td></tr>"
        )
    col_info_html = (
        '<table>'
        '<thead><tr>'
        '<th>컬럼명</th><th>타입</th><th>분류</th>'
        '<th>결측치</th>'
        '<th>고유값</th>'
        '</tr></thead>'
        '<tbody>' + ''.join(col_rows) + '</tbody></table>'
    )

    # 기초 통계량 (숫자형)
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc = desc.round(2)
        stat_rows = []
        for idx, row in desc.iterrows():
            stat_rows.append(
                f"<tr><td>{idx}</td>"
                f"<td >{row['count']:,.0f}</td>"
                f"<td >{row['mean']:,.2f}</td>"
                f"<td >{row['std']:,.2f}</td>"
                f"<td >{row['min']:,.2f}</td>"
                f"<td >{row['25%']:,.2f}</td>"
                f"<td >{row['50%']:,.2f}</td>"
                f"<td >{row['75%']:,.2f}</td>"
                f"<td >{row['max']:,.2f}</td></tr>"
            )
        stats_html = (
            '<table>'
            '<thead><tr><th>컬럼</th>'
            '<th>count</th>'
            '<th>mean</th>'
            '<th>std</th>'
            '<th>min</th>'
            '<th>25%</th>'
            '<th>50%</th>'
            '<th>75%</th>'
            '<th>max</th>'
            '</tr></thead>'
            '<tbody>' + ''.join(stat_rows) + '</tbody></table>'
        )
    else:
        stats_html = ""

    sep = '<div style="height:16px"></div>'
    return overview_html + sep + col_info_html + sep + stats_html


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


def _generate_histograms(
    df: pd.DataFrame, numeric_cols: list[str], job_dir: str
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
            fig = plotly_histogram(series.tolist(), title=title, xlabel=col)
            plotly_save_png(fig, os.path.join(job_dir, f"히스토그램_{col}.png"))
            charts.append({
                "title": title,
                "plotly": json.loads(plotly_to_json(fig)),
            })
        except Exception:
            logger.warning("히스토그램 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_bar_charts(
    df: pd.DataFrame, cat_cols: list[str], job_dir: str
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
            fig = plotly_bar_h(labels, values, title=title)
            plotly_save_png(fig, os.path.join(job_dir, f"바차트_{col}.png"),
                            height=max(400, len(labels) * 50))
            charts.append({
                "title": title,
                "plotly": json.loads(plotly_to_json(fig)),
            })
        except Exception:
            logger.warning("바 차트 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_line_charts(
    df: pd.DataFrame, dt_cols: list[str], job_dir: str
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
            fig = plotly_line(x, y, title=title, xlabel="날짜", ylabel="건수")
            plotly_save_png(fig, os.path.join(job_dir, f"시간추이_{col}.png"))
            charts.append({
                "title": title,
                "plotly": json.loads(plotly_to_json(fig)),
            })
        except Exception:
            logger.warning("라인 차트 생성 실패: %s\n%s", col, traceback.format_exc())
    return charts


def _generate_heatmap(
    df: pd.DataFrame, numeric_cols: list[str], job_dir: str
) -> list[dict]:
    """상관관계 히트맵 (숫자 컬럼 2개 이상일 때)."""
    if len(numeric_cols) < 2:
        return []
    charts = []
    try:
        corr = df[numeric_cols].corr()
        title = "상관관계 히트맵"
        fig = plotly_heatmap(
            z=corr.values.tolist(),
            x_labels=corr.columns.tolist(),
            y_labels=corr.index.tolist(),
            title=title,
        )
        plotly_save_png(fig, os.path.join(job_dir, "상관관계_히트맵.png"))
        charts.append({
            "title": title,
            "plotly": json.loads(plotly_to_json(fig)),
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
    df: pd.DataFrame, numeric_cols: list[str], job_dir: str
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
                f"<b>{col}</b>: 이상치 {n_out:,}건 ({pct:.1f}%) "
                f"[정상 범위: {lower:,.2f} ~ {upper:,.2f}]"
            )
            fig = go.Figure(go.Box(y=series.tolist(), name=col,
                                   marker_color="#3498DB",
                                   boxmean="sd"))
            _apply_style(fig,
                title=dict(text=title, font=dict(size=16)),
                yaxis_title=col,
                margin=dict(t=60, b=30, l=60, r=30),
            )
            plotly_save_png(fig, os.path.join(job_dir, f"박스플롯_{col}.png"))
            charts.append({
                "title": title,
                "plotly": json.loads(plotly_to_json(fig)),
            })
        except Exception:
            logger.warning("박스 플롯 생성 실패: %s\n%s", col, traceback.format_exc())

    # 이상치가 없는 경우
    if not details_lines:
        # 전체 숫자 컬럼에 대해서도 이상치 없음 표시
        for col, n_out, lower, upper in outlier_info:
            details_lines.append(
                f"<b>{col}</b>: 이상치 없음 "
                f"[정상 범위: {lower:,.2f} ~ {upper:,.2f}]"
            )

    # 아코디언 형태로 이상치 결과 표시
    acc_items = ""
    all_info = selected if selected else outlier_info
    for idx, (col, n_out, lower, upper) in enumerate(all_info):
        pct = n_out / len(df) * 100 if len(df) > 0 else 0
        status = f"이상치 {n_out:,}건 ({pct:.1f}%)" if n_out > 0 else "이상치 없음"

        # 이상치 예시 데이터
        examples_html = ""
        if n_out > 0:
            mask, _, _ = _detect_outliers(df, col)
            outlier_vals = df.loc[mask, col].dropna().head(5).tolist()
            examples = ", ".join([f"{v:,.2f}" for v in outlier_vals])
            examples_html = (
                f'<div class="acc-review">'
                f'이상치 예시 (최대 5건): {examples}</div>'
            )

        body = (
            f'<div class="acc-review">'
            f'정상 범위 (IQR): {lower:,.2f} ~ {upper:,.2f}</div>'
            f'<div class="acc-review">'
            f'이상치: {n_out:,}건 / 전체 {len(df):,}건 ({pct:.1f}%)</div>'
            f'{examples_html}'
        )
        open_cls = " open" if idx == 0 else ""
        acc_items += (
            f'<div class="acc-item">'
            f'<button class="acc-btn{open_cls}" onclick="toggleAcc(this)">'
            f'{col} - {status}</button>'
            f'<div class="acc-body{open_cls}">{body}</div></div>'
        )

    # 결측치 요약도 추가
    missing_cols = [(c, int(df[c].isnull().sum())) for c in df.columns if df[c].isnull().sum() > 0]
    if missing_cols:
        missing_rows = ""
        for col, cnt in sorted(missing_cols, key=lambda x: x[1], reverse=True):
            pct = cnt / len(df) * 100
            missing_rows += (
                f'<div class="acc-review">'
                f'{col}: {cnt:,}건 ({pct:.1f}%)</div>'
            )
        acc_items += (
            f'<div class="acc-item">'
            f'<button class="acc-btn" onclick="toggleAcc(this)">'
            f'결측치 현황 ({len(missing_cols)}개 컬럼)</button>'
            f'<div class="acc-body">{missing_rows}</div></div>'
        )

    if not all_info and not missing_cols:
        acc_items = "<p>이상치와 결측치가 없습니다.</p>"

    return charts, acc_items


# ---------------------------------------------------------------------------
# 메인 EDA 실행
# ---------------------------------------------------------------------------

def run_eda(
    df: pd.DataFrame,
    job_id: str,
    chart_mode: str = "plotly",
) -> dict:
    """
    EDA 실행 -> dict 반환.

    Returns
    -------
    dict with keys: summary_html, charts, details_html, downloads
    """
    # 결과 저장 디렉토리
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # -- 1. 컬럼 타입 탐지 --
    col_types = detect_column_types(df)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    dt_cols = [c for c, t in col_types.items() if t == "datetime"]

    # -- 2. Summary HTML --
    summary_html = _build_summary_html(df, col_types)

    # -- 3. 차트 생성 --
    charts: list[dict] = []

    # 히스토그램
    try:
        charts.extend(_generate_histograms(df, numeric_cols, job_dir))
    except Exception:
        logger.warning("히스토그램 생성 중 오류\n%s", traceback.format_exc())

    # 범주형 바 차트
    try:
        charts.extend(_generate_bar_charts(df, cat_cols, job_dir))
    except Exception:
        logger.warning("바 차트 생성 중 오류\n%s", traceback.format_exc())

    # 시간 추이
    try:
        charts.extend(_generate_line_charts(df, dt_cols, job_dir))
    except Exception:
        logger.warning("라인 차트 생성 중 오류\n%s", traceback.format_exc())

    # 상관관계 히트맵
    try:
        charts.extend(_generate_heatmap(df, numeric_cols, job_dir))
    except Exception:
        logger.warning("히트맵 생성 중 오류\n%s", traceback.format_exc())

    # 박스 플롯 + 이상치 요약
    try:
        box_charts, details_html = _generate_box_plots(df, numeric_cols, job_dir)
        charts.extend(box_charts)
    except Exception:
        logger.warning("박스 플롯 생성 중 오류\n%s", traceback.format_exc())
        details_html = "<p>이상치 분석 중 오류가 발생했습니다.</p>"

    # -- 4. CSV 저장 --
    # 원본 데이터
    raw_filename = "원본_데이터.csv"
    try:
        df.to_csv(os.path.join(job_dir, raw_filename), index=False, encoding="utf-8-sig")
    except Exception:
        logger.warning("원본 CSV 저장 실패\n%s", traceback.format_exc())

    # 기초 통계 요약
    stats_filename = "기초_통계량.csv"
    try:
        if numeric_cols:
            desc = df[numeric_cols].describe().T.round(2)
            desc.index.name = "컬럼"
            desc.to_csv(os.path.join(job_dir, stats_filename), encoding="utf-8-sig")
    except Exception:
        logger.warning("통계 CSV 저장 실패\n%s", traceback.format_exc())

    # 컬럼 정보
    colinfo_filename = "컬럼_정보.csv"
    try:
        col_info_rows = []
        for col in df.columns:
            col_info_rows.append({
                "컬럼명": col,
                "타입": str(df[col].dtype),
                "분류": col_types.get(col, "-"),
                "결측치": int(df[col].isnull().sum()),
                "고유값": int(df[col].nunique()),
            })
        pd.DataFrame(col_info_rows).to_csv(os.path.join(job_dir, colinfo_filename), index=False, encoding="utf-8-sig")
    except Exception:
        logger.warning("컬럼정보 CSV 저장 실패\n%s", traceback.format_exc())

    # -- 5. 다운로드 --
    chart_zip = bundle_zip(job_dir, "차트_이미지.zip", pattern="*.png")
    downloads = [
        {"filename": raw_filename, "label": "원본 데이터 CSV"},
        {"filename": stats_filename, "label": "기초 통계량 CSV"},
        {"filename": colinfo_filename, "label": "컬럼 정보 CSV"},
        {"filename": chart_zip, "label": "차트 이미지 ZIP"},
    ]

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
