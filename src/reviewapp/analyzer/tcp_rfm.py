"""TCP/RFM 분석 모듈 - 시간(Time)·고객(Customer)·제품(Product) + RFM 세그먼트."""
import os
import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils

from reviewapp.analyzer.chart_utils import (
    plotly_donut,
    plotly_bar_h,
    plotly_line,
    plotly_heatmap,
    plotly_to_json,
    plotly_save_png,
    bundle_zip,
    _apply_style,
    CHART_COLORS,
)

logger = logging.getLogger("review-analyzer")

WORK_DIR = os.path.join(os.path.expanduser("~"), ".review-analyzer")
RESULT_DIR = os.path.join(WORK_DIR, "results")

# ---------------------------------------------------------------------------
# 요일 한글 매핑
# ---------------------------------------------------------------------------
DAY_KR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

# ---------------------------------------------------------------------------
# RFM 세그먼트 정의
# ---------------------------------------------------------------------------
SEGMENTS = {
    "챔피언": lambda r, f, m: r >= 4 and f >= 4 and m >= 4,
    "충성 고객": lambda r, f, m: f >= 3 and m >= 3,
    "잠재 충성": lambda r, f, m: r >= 3 and f >= 2,
    "신규 고객": lambda r, f, m: r >= 4 and f <= 1,
    "이탈 위험": lambda r, f, m: r <= 2 and f >= 2,
    "이탈 고객": lambda r, f, m: r <= 2 and f <= 2,
}

SEGMENT_COLORS = {
    "챔피언": "#22c55e",
    "충성 고객": "#3b82f6",
    "잠재 충성": "#8b5cf6",
    "신규 고객": "#06b6d4",
    "이탈 위험": "#f59e0b",
    "이탈 고객": "#ef4444",
    "일반 고객": "#94a3b8",
}


# ---------------------------------------------------------------------------
# 컬럼 자동 탐지
# ---------------------------------------------------------------------------
def detect_commerce_columns(df: pd.DataFrame) -> dict:
    """이커머스 데이터에서 날짜·고객·상품·수량·단가 컬럼을 자동 탐지한다."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["date", "날짜", "일자", "invoicedate", "orderdate", "order_date"]):
            mapping.setdefault("date", col)
        elif cl in ("invoice", "invoiceno", "invoice_no") and "date" not in mapping:
            pass  # InvoiceNo는 날짜가 아님 - 건너뜀
        elif any(k in cl for k in ["customer", "고객", "회원", "user"]):
            mapping.setdefault("customer", col)
        elif any(k in cl for k in ["product", "상품", "제품", "item", "description", "stock"]):
            mapping.setdefault("product", col)
        elif any(k in cl for k in ["quantity", "수량", "qty"]):
            mapping.setdefault("quantity", col)
        elif any(k in cl for k in ["price", "가격", "단가", "unit"]):
            mapping.setdefault("price", col)
    return mapping


# ---------------------------------------------------------------------------
# RFM 점수 계산 헬퍼
# ---------------------------------------------------------------------------
def _safe_qcut(series: pd.Series, q: int, labels) -> pd.Series:
    """pd.qcut 에서 중복 bin edge 발생 시 rank 기반으로 폴백한다."""
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        # 값이 적어서 q 분위로 나눌 수 없는 경우 rank 기반 처리
        ranks = series.rank(method="first")
        try:
            return pd.qcut(ranks, q=q, labels=labels[:len(pd.qcut(ranks, q=q, duplicates="drop").cat.categories)], duplicates="drop")
        except Exception:
            # 최후 수단: 전부 중간 점수
            return pd.Series([3] * len(series), index=series.index, dtype=int)


def _assign_segment(row) -> str:
    """R/F/M 점수로 세그먼트를 결정한다."""
    r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])
    for name, rule in SEGMENTS.items():
        if rule(r, f, m):
            return name
    return "일반 고객"


# ---------------------------------------------------------------------------
# 메인 분석 함수
# ---------------------------------------------------------------------------
def run_tcp(
    df: pd.DataFrame,
    job_id: str,
    chart_mode: str = "plotly",
    col_map: dict | None = None,
    dimensions: list | None = None,
) -> dict:
    """TCP/RFM 분석 실행.

    Returns:
        dict with keys: summary_html, charts, details_html, downloads
    """
    if dimensions is None:
        dimensions = ["time", "customer", "product"]

    charts: list[dict] = []
    downloads: list[dict] = []
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    result_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(result_dir, exist_ok=True)

    # ----- 컬럼 매핑 결정 -------------------------------------------------
    if col_map is None:
        col_map = {}
    # 빈 문자열인 값 제거
    col_map = {k: v for k, v in col_map.items() if v}

    auto_detected = detect_commerce_columns(df)
    # col_map 에 없는 항목은 auto_detected 로 보완
    for key in ["date", "customer", "product", "quantity", "price"]:
        if key not in col_map and key in auto_detected:
            col_map[key] = auto_detected[key]

    # 필수 컬럼 확인
    missing_keys = [k for k in ["date", "customer", "quantity", "price"] if k not in col_map]
    if missing_keys:
        missing_names = {"date": "날짜", "customer": "고객", "quantity": "수량", "price": "단가"}
        msg = ", ".join(f"{missing_names.get(k, k)}" for k in missing_keys)
        raise ValueError(f"필수 컬럼을 찾을 수 없습니다: {msg}. 컬럼 매핑을 확인해주세요.")

    date_col = col_map["date"]
    cust_col = col_map["customer"]
    prod_col = col_map.get("product")  # 없으면 상품 분석 스킵
    qty_col = col_map["quantity"]
    price_col = col_map["price"]

    # ----- 데이터 전처리 ---------------------------------------------------
    # 날짜 변환
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, cust_col, qty_col, price_col])

    # 숫자 변환
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[qty_col, price_col])

    # 음수 제거
    df = df[(df[qty_col] > 0) & (df[price_col] > 0)].copy()

    if len(df) == 0:
        raise ValueError("유효한 데이터가 없습니다. 수량/단가가 양수이고 날짜가 올바른지 확인해주세요.")

    # 매출 컬럼
    df["매출"] = df[qty_col] * df[price_col]

    # ----- 요약 HTML -------------------------------------------------------
    total_revenue = df["매출"].sum()
    total_orders = len(df)
    unique_customers = df[cust_col].nunique()
    unique_products = df[prod_col].nunique() if prod_col else "N/A"
    date_min = df[date_col].min().strftime("%Y-%m-%d")
    date_max = df[date_col].max().strftime("%Y-%m-%d")

    col_map_display = (
        f"날짜={date_col}, 고객={cust_col}, "
        f"상품={prod_col or '(미지정)'}, 수량={qty_col}, 단가={price_col}"
    )

    summary_html = f"""
    <table>
        <thead><tr><th>항목</th><th>값</th></tr></thead>
        <tbody>
            <tr><td>분석 기간</td><td>{date_min} ~ {date_max}</td></tr>
            <tr><td>총 거래 건수</td><td>{total_orders:,}건</td></tr>
            <tr><td>고유 고객 수</td><td>{unique_customers:,}명</td></tr>
            <tr><td>고유 상품 수</td><td>{unique_products if isinstance(unique_products, str) else f'{unique_products:,}개'}</td></tr>
            <tr><td>총 매출</td><td>{total_revenue:,.0f}원</td></tr>
            <tr><td>컬럼 매핑</td><td>{col_map_display}</td></tr>
        </tbody>
    </table>
    """

    # =====================================================================
    # 1. 시간 분석 (Time Analysis)
    # =====================================================================
    if "time" in dimensions:
        # --- 1-1. 월별 매출 추이 ---
        df["연월"] = df[date_col].dt.to_period("M").astype(str)
        monthly = df.groupby("연월", sort=True)["매출"].sum().reset_index()
        monthly.columns = ["연월", "매출"]

        fig_monthly = plotly_line(
            x=monthly["연월"].tolist(),
            y=monthly["매출"].tolist(),
            title="월별 매출 추이",
            xlabel="월",
            ylabel="매출(원)",
        )
        plotly_save_png(fig_monthly, os.path.join(result_dir, "월별_매출_추이.png"))
        charts.append({"title": "월별 매출 추이", "plotly": json.loads(plotly_to_json(fig_monthly))})

        # CSV 저장
        monthly_path = os.path.join(job_dir, "TCP_매출분석.csv")
        monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        downloads.append({"filename": "TCP_매출분석.csv", "label": "월별 매출 분석 CSV"})

        # --- 1-2. 일별 주문 건수 ---
        df["날짜_only"] = df[date_col].dt.date
        daily_orders = df.groupby("날짜_only").size().reset_index(name="주문건수")
        daily_orders["날짜_only"] = daily_orders["날짜_only"].astype(str)

        fig_daily = plotly_line(
            x=daily_orders["날짜_only"].tolist(),
            y=daily_orders["주문건수"].tolist(),
            title="일별 주문 건수",
            xlabel="날짜",
            ylabel="주문 건수",
        )
        plotly_save_png(fig_daily, os.path.join(result_dir, "일별_주문_건수.png"))
        charts.append({"title": "일별 주문 건수", "plotly": json.loads(plotly_to_json(fig_daily))})

        # --- 1-3. 요일별 주문 비율 ---
        df["요일"] = df[date_col].dt.dayofweek.map(DAY_KR)
        dow_counts = df["요일"].value_counts()
        dow_sorted = [dow_counts.get(d, 0) for d in DAY_ORDER]

        fig_dow = go.Figure(go.Bar(
            x=DAY_ORDER,
            y=dow_sorted,
            text=[f"{v:,}건" for v in dow_sorted],
            textposition="outside",
            marker_color=["#ef4444" if d in ("토", "일") else "#4361ee" for d in DAY_ORDER],
        ))
        _apply_style(fig_dow,
            title=dict(text="요일별 주문 비율", font=dict(size=14, color="#333"), x=0.5, xanchor="center"),
            xaxis_title="요일",
            yaxis_title="주문 건수",
        )
        plotly_save_png(fig_dow, os.path.join(result_dir, "요일별_주문_비율.png"))
        charts.append({"title": "요일별 주문 비율", "plotly": json.loads(plotly_to_json(fig_dow))})

    # =====================================================================
    # 2. 상품 분석 (Product Analysis)
    # =====================================================================
    if "product" in dimensions and prod_col and prod_col in df.columns:
        # --- 2-1. 상품별 매출 TOP 15 ---
        prod_revenue = (
            df.groupby(prod_col)["매출"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        fig_prod_rev = plotly_bar_h(
            labels=prod_revenue.index.tolist()[::-1],
            values=prod_revenue.values.tolist()[::-1],
            title="상품별 매출 TOP 15",
            color="#6366f1",
            text_suffix="원",
        )
        fig_prod_rev.update_layout(xaxis_title="매출(원)", yaxis=dict(autorange="reversed"))
        plotly_save_png(fig_prod_rev, os.path.join(result_dir, "상품별_매출_TOP15.png"))
        charts.append({"title": "상품별 매출 TOP 15", "plotly": json.loads(plotly_to_json(fig_prod_rev))})

        # --- 2-2. 상품별 판매량 TOP 15 ---
        prod_qty = (
            df.groupby(prod_col)[qty_col]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        fig_prod_qty = plotly_bar_h(
            labels=prod_qty.index.tolist()[::-1],
            values=prod_qty.values.tolist()[::-1],
            title="상품별 판매량 TOP 15",
            color="#f59e0b",
            text_suffix="개",
        )
        fig_prod_qty.update_layout(xaxis_title="판매량(개)", yaxis=dict(autorange="reversed"))
        plotly_save_png(fig_prod_qty, os.path.join(result_dir, "상품별_판매량_TOP15.png"))
        charts.append({"title": "상품별 판매량 TOP 15", "plotly": json.loads(plotly_to_json(fig_prod_qty))})

    # =====================================================================
    # 3. RFM 분석 (고객 기준)
    # =====================================================================
    if "customer" not in dimensions:
        # ZIP 번들 (PNG 차트)
        chart_zip = bundle_zip(result_dir, "차트_이미지.zip", pattern="*.png")
        downloads.append({"filename": chart_zip, "label": "차트 이미지 ZIP"})
        return {
            "summary_html": summary_html,
            "charts": charts,
            "details_html": "",
            "downloads": downloads,
        }
    now = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(cust_col).agg(
        R=(date_col, lambda x: (now - x.max()).days),
        F=(date_col, "count"),
        M=("매출", "sum"),
    ).reset_index()

    # 점수 부여 (1~5)
    rfm["R_score"] = _safe_qcut(rfm["R"], 5, labels=[5, 4, 3, 2, 1])  # R 은 낮을수록 좋으므로 역순
    rfm["F_score"] = _safe_qcut(rfm["F"], 5, labels=[1, 2, 3, 4, 5])
    rfm["M_score"] = _safe_qcut(rfm["M"], 5, labels=[1, 2, 3, 4, 5])

    # 숫자로 변환 (Categorical -> int)
    for sc in ["R_score", "F_score", "M_score"]:
        rfm[sc] = rfm[sc].astype(int)

    # 세그먼트 부여
    rfm["세그먼트"] = rfm.apply(_assign_segment, axis=1)

    # CSV 저장
    rfm_path = os.path.join(job_dir, "RFM_결과.csv")
    rfm.to_csv(rfm_path, index=False, encoding="utf-8-sig")
    downloads.append({"filename": "RFM_결과.csv", "label": "RFM 분석 결과 CSV"})

    # --- 3-1. 고객 세그먼트 분포 (도넛) ---
    seg_counts = rfm["세그먼트"].value_counts()
    seg_labels = seg_counts.index.tolist()
    seg_values = seg_counts.values.tolist()
    seg_colors = [SEGMENT_COLORS.get(s, "#94a3b8") for s in seg_labels]

    fig_seg_donut = plotly_donut(
        labels=seg_labels,
        values=seg_values,
        title="고객 세그먼트 분포",
        colors=seg_colors,
    )
    plotly_save_png(fig_seg_donut, os.path.join(result_dir, "고객_세그먼트_분포.png"))
    charts.append({"title": "고객 세그먼트 분포", "plotly": json.loads(plotly_to_json(fig_seg_donut))})

    # --- 3-2. 세그먼트별 평균 RFM (그룹 바 차트) ---
    seg_avg = rfm.groupby("세그먼트")[["R_score", "F_score", "M_score"]].mean()
    # 세그먼트 순서 고정
    all_segments = list(SEGMENTS.keys()) + ["일반 고객"]
    seg_avg = seg_avg.reindex([s for s in all_segments if s in seg_avg.index])

    fig_seg_rfm = go.Figure()
    for col_name, display_name, color in [
        ("R_score", "R (최근성)", "#22c55e"),
        ("F_score", "F (빈도)", "#3b82f6"),
        ("M_score", "M (금액)", "#f59e0b"),
    ]:
        fig_seg_rfm.add_trace(go.Bar(
            name=display_name,
            x=seg_avg.index.tolist(),
            y=seg_avg[col_name].tolist(),
            marker_color=color,
            text=[f"{v:.1f}" for v in seg_avg[col_name]],
            textposition="outside",
        ))
    _apply_style(fig_seg_rfm,
        barmode="group",
        title=dict(text="세그먼트별 평균 RFM", font=dict(size=14, color="#333"), x=0.5, xanchor="center"),
        xaxis_title="세그먼트",
        yaxis_title="평균 점수",
        legend=dict(orientation="h", y=-0.2),
    )
    plotly_save_png(fig_seg_rfm, os.path.join(result_dir, "세그먼트별_평균_RFM.png"))
    charts.append({"title": "세그먼트별 평균 RFM", "plotly": json.loads(plotly_to_json(fig_seg_rfm))})


    # =====================================================================
    # 상세 HTML: RFM 세그먼트 테이블 + 인사이트
    # =====================================================================
    # 세그먼트 요약 테이블
    seg_summary = rfm.groupby("세그먼트").agg(
        고객수=(cust_col, "count"),
        평균R=("R", "mean"),
        평균F=("F", "mean"),
        평균M=("M", "mean"),
    ).reset_index()
    seg_summary = seg_summary.sort_values("고객수", ascending=False)

    seg_table_rows = ""
    for _, row in seg_summary.iterrows():
        seg_name = row["세그먼트"]
        color = SEGMENT_COLORS.get(seg_name, "#94a3b8")
        pct = row["고객수"] / len(rfm) * 100
        seg_table_rows += f"""
        <tr>
            <td><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};margin-right:6px;"></span>{seg_name}</td>
            <td>{int(row['고객수']):,}명 ({pct:.1f}%)</td>
            <td>{row['평균R']:.1f}일</td>
            <td>{row['평균F']:.1f}회</td>
            <td>{row['평균M']:,.0f}원</td>
        </tr>"""

    # 인사이트 생성
    insights = []
    champion_pct = seg_summary.loc[seg_summary["세그먼트"] == "챔피언", "고객수"].sum() / len(rfm) * 100 if "챔피언" in seg_summary["세그먼트"].values else 0
    churn_pct = seg_summary.loc[seg_summary["세그먼트"] == "이탈 고객", "고객수"].sum() / len(rfm) * 100 if "이탈 고객" in seg_summary["세그먼트"].values else 0
    risk_pct = seg_summary.loc[seg_summary["세그먼트"] == "이탈 위험", "고객수"].sum() / len(rfm) * 100 if "이탈 위험" in seg_summary["세그먼트"].values else 0

    if champion_pct > 0:
        insights.append(f"챔피언 고객이 전체의 <strong>{champion_pct:.1f}%</strong>를 차지합니다. 이 고객군에 대한 VIP 혜택을 강화하세요.")
    if churn_pct > 20:
        insights.append(f"이탈 고객 비율이 <strong>{churn_pct:.1f}%</strong>로 높습니다. 재활성화 캠페인을 고려해보세요.")
    if risk_pct > 15:
        insights.append(f"이탈 위험 고객이 <strong>{risk_pct:.1f}%</strong>입니다. 할인 쿠폰이나 맞춤 추천으로 이탈을 방지하세요.")

    top_seg = seg_summary.iloc[0]["세그먼트"] if len(seg_summary) > 0 else "N/A"
    insights.append(f"가장 많은 고객 세그먼트는 <strong>{top_seg}</strong>입니다.")

    avg_order_value = total_revenue / total_orders
    insights.append(f"건당 평균 주문 금액은 <strong>{avg_order_value:,.0f}원</strong>입니다.")

    insights_html = "".join(f"<li style='margin-bottom:4px;'>{i}</li>" for i in insights)

    details_html = f"""
    <table>
        <thead>
            <tr><th>세그먼트</th><th>고객 수</th><th>평균 R (최근성)</th><th>평균 F (빈도)</th><th>평균 M (금액)</th></tr>
        </thead>
        <tbody>{seg_table_rows}</tbody>
    </table>
    <div style="height:12px"></div>
    <ul>{insights_html}</ul>
    <div style="color:#888;margin-top:8px;">
        R(Recency): 마지막 구매 후 경과 일수 / F(Frequency): 구매 횟수 / M(Monetary): 총 구매 금액<br>
        각 지표는 5분위수(qcut)로 1~5점 부여 (5점이 가장 우수)
    </div>
    """

    # --- ZIP 번들 (PNG 차트) ---
    chart_zip = bundle_zip(result_dir, "차트_이미지.zip", pattern="*.png")
    downloads.append({"filename": chart_zip, "label": "차트 이미지 ZIP"})

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
