"""감성 분석 모듈 - 사전 기반 한국어 감성 분석."""
import os
import json
import logging
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import detect_text_column
from analyzer.chart_utils import (
    get_korean_font_path,
    setup_matplotlib_korean,
    plotly_donut,
    plotly_to_json,
    fig_to_base64,
)

logger = logging.getLogger("review-analyzer")

# ---------------------------------------------------------------------------
# 사전 로드
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LEXICON_PATH = os.path.join(_BASE_DIR, "data", "sentiment_lexicon.json")


def _load_lexicon() -> dict:
    with open(_LEXICON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 감성 점수 계산
# ---------------------------------------------------------------------------

def _score_review(text: str, lexicon: dict) -> tuple[int, list[str]]:
    """리뷰 텍스트에 대해 감성 점수와 매칭된 키워드 목록을 반환한다.

    긴 키워드(멀티토큰)부터 먼저 매칭하여 중복 카운트를 방지한다.
    """
    if not isinstance(text, str) or not text.strip():
        return 0, []

    score = 0
    matched = []
    # 긴 키워드부터 매칭 (예: "넘 좋" 을 "좋" 보다 먼저)
    sorted_keys = sorted(lexicon.keys(), key=len, reverse=True)
    for word in sorted_keys:
        if word in text:
            score += lexicon[word]
            matched.append(word)
    return score, matched


def _classify_sentiment(score: int) -> str:
    if score >= 1:
        return "긍정"
    elif score <= -1:
        return "부정"
    return "중립"


# ---------------------------------------------------------------------------
# 차트 생성
# ---------------------------------------------------------------------------

def _build_donut_chart(counts: dict, chart_mode: str) -> dict:
    """감성 비율 도넛 차트."""
    labels = ["긍정", "중립", "부정"]
    values = [counts.get(l, 0) for l in labels]
    colors = ["#10B981", "#FBBF24", "#EF4444"]  # green, yellow, red

    fig = plotly_donut(labels, values, title="감성 분석 비율", colors=colors)

    if chart_mode == "plotly":
        return {"title": "감성 분석 비율", "plotly": json.loads(plotly_to_json(fig))}
    else:
        # base64 fallback - matplotlib donut
        setup_matplotlib_korean()
        mfig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct="%1.1f%%", pctdistance=0.75, startangle=90,
        )
        centre_circle = plt.Circle((0, 0), 0.5, fc="white")
        ax.add_artist(centre_circle)
        ax.set_title("감성 분석 비율", fontsize=14)
        return {"title": "감성 분석 비율", "image": fig_to_base64(mfig)}


def _build_wordcloud_chart(keyword_freq: dict, keyword_scores: dict) -> dict:
    """감성 키워드 워드클라우드 (항상 base64 이미지로 반환).

    Args:
        keyword_freq: {word: occurrence_count} - 워드클라우드 크기 결정
        keyword_scores: {word: lexicon_score} - 워드클라우드 색상 결정
    """
    if not keyword_freq:
        return None

    font_path = get_korean_font_path()

    # 워드클라우드 불가 시 바 차트 대체
    if font_path is None:
        return _build_keyword_bar_fallback(keyword_scores)

    try:
        from wordcloud import WordCloud
    except ImportError:
        return _build_keyword_bar_fallback(keyword_scores)

    # 색상 함수
    score_map = keyword_scores

    def color_func(word, **kwargs):
        s = score_map.get(word, 0)
        if s >= 2:
            return "#059669"
        elif s >= 1:
            return "#10B981"
        elif s <= -2:
            return "#DC2626"
        elif s <= -1:
            return "#EF4444"
        return "#6B7280"  # 중립 (회색)

    setup_matplotlib_korean()
    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        max_words=100,
        color_func=color_func,
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(keyword_freq)

    mfig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("감성 키워드 워드클라우드", fontsize=14)

    return {"title": "감성 키워드 워드클라우드", "image": fig_to_base64(mfig)}


def _build_keyword_bar_fallback(keyword_scores: dict) -> dict:
    """워드클라우드를 사용할 수 없을 때 바 차트로 대체."""
    pos = {w: s for w, s in keyword_scores.items() if s > 0}
    neg = {w: abs(s) for w, s in keyword_scores.items() if s < 0}

    pos_top = sorted(pos.items(), key=lambda x: x[1], reverse=True)[:15]
    neg_top = sorted(neg.items(), key=lambda x: x[1], reverse=True)[:15]

    setup_matplotlib_korean()
    mfig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if pos_top:
        words, vals = zip(*pos_top)
        axes[0].barh(range(len(words)), vals, color="#10B981")
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words)
        axes[0].invert_yaxis()
        axes[0].set_title("긍정 키워드", fontsize=13)
    else:
        axes[0].text(0.5, 0.5, "긍정 키워드 없음", ha="center", va="center", fontsize=12)
        axes[0].set_title("긍정 키워드", fontsize=13)

    if neg_top:
        words, vals = zip(*neg_top)
        axes[1].barh(range(len(words)), vals, color="#EF4444")
        axes[1].set_yticks(range(len(words)))
        axes[1].set_yticklabels(words)
        axes[1].invert_yaxis()
        axes[1].set_title("부정 키워드", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "부정 키워드 없음", ha="center", va="center", fontsize=12)
        axes[1].set_title("부정 키워드", fontsize=13)

    mfig.suptitle("감성 키워드 빈도 (워드클라우드 대체)", fontsize=14)
    mfig.tight_layout()
    return {"title": "감성 키워드 워드클라우드", "image": fig_to_base64(mfig)}


def _build_top_keywords_chart(keyword_counter_pos: Counter, keyword_counter_neg: Counter,
                              chart_mode: str) -> dict:
    """긍정 TOP5 + 부정 TOP5 바 차트 (side-by-side subplots)."""
    pos_top5 = keyword_counter_pos.most_common(5)
    neg_top5 = keyword_counter_neg.most_common(5)

    # 빈 경우 패딩
    if not pos_top5:
        pos_top5 = [("(없음)", 0)]
    if not neg_top5:
        neg_top5 = [("(없음)", 0)]

    pos_words = [w for w, _ in pos_top5]
    pos_counts = [c for _, c in pos_top5]
    neg_words = [w for w, _ in neg_top5]
    neg_counts = [c for _, c in neg_top5]

    if chart_mode == "plotly":
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["긍정 키워드 TOP 5", "부정 키워드 TOP 5"],
                            horizontal_spacing=0.15)

        fig.add_trace(
            go.Bar(
                y=pos_words, x=pos_counts, orientation="h",
                marker_color="#10B981",
                text=[f"{c}건" for c in pos_counts],
                textposition="outside",
                name="긍정",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                y=neg_words, x=neg_counts, orientation="h",
                marker_color="#EF4444",
                text=[f"{c}건" for c in neg_counts],
                textposition="outside",
                name="부정",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title=dict(text="긍정 / 부정 키워드 TOP 5", font=dict(size=16)),
            showlegend=False,
            margin=dict(t=80, b=30, l=100, r=60),
            height=400,
        )
        # y축 순서: 높은 건수가 위쪽
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_xaxes(title_text="건수", row=1, col=1)
        fig.update_xaxes(title_text="건수", row=1, col=2)

        return {"title": "긍정 / 부정 키워드 TOP 5", "plotly": json.loads(plotly_to_json(fig))}
    else:
        setup_matplotlib_korean()
        mfig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].barh(range(len(pos_words)), pos_counts, color="#10B981")
        axes[0].set_yticks(range(len(pos_words)))
        axes[0].set_yticklabels(pos_words)
        axes[0].invert_yaxis()
        axes[0].set_title("긍정 키워드 TOP 5", fontsize=13)
        axes[0].set_xlabel("건수")

        axes[1].barh(range(len(neg_words)), neg_counts, color="#EF4444")
        axes[1].set_yticks(range(len(neg_words)))
        axes[1].set_yticklabels(neg_words)
        axes[1].invert_yaxis()
        axes[1].set_title("부정 키워드 TOP 5", fontsize=13)
        axes[1].set_xlabel("건수")

        mfig.suptitle("긍정 / 부정 키워드 TOP 5", fontsize=14)
        mfig.tight_layout()
        return {"title": "긍정 / 부정 키워드 TOP 5", "image": fig_to_base64(mfig)}


# ---------------------------------------------------------------------------
# HTML 생성
# ---------------------------------------------------------------------------

def _build_summary_html(counts: dict, total: int, avg_score: float) -> str:
    """요약 HTML 테이블."""
    rows = ""
    for label, color in [("긍정", "#10B981"), ("중립", "#FBBF24"), ("부정", "#EF4444")]:
        cnt = counts.get(label, 0)
        pct = cnt / total * 100 if total > 0 else 0
        rows += (
            f"<tr>"
            f"<td><span style='display:inline-block;width:12px;height:12px;"
            f"border-radius:50%;background:{color};margin-right:6px;'></span>{label}</td>"
            f"<td style='text-align:right;'>{cnt:,}건</td>"
            f"<td style='text-align:right;'>{pct:.1f}%</td>"
            f"</tr>"
        )

    return (
        f"<table style='width:100%;border-collapse:collapse;margin-bottom:12px;'>"
        f"<thead><tr style='border-bottom:2px solid #e5e7eb;'>"
        f"<th style='text-align:left;padding:8px;'>감성</th>"
        f"<th style='text-align:right;padding:8px;'>건수</th>"
        f"<th style='text-align:right;padding:8px;'>비율</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"<tfoot><tr style='border-top:2px solid #e5e7eb;font-weight:bold;'>"
        f"<td style='padding:8px;'>합계</td>"
        f"<td style='text-align:right;padding:8px;'>{total:,}건</td>"
        f"<td style='text-align:right;padding:8px;'>평균 점수: {avg_score:.2f}</td>"
        f"</tr></tfoot>"
        f"</table>"
    )


def _build_details_html(df: pd.DataFrame, text_col: str) -> str:
    """감성별 대표 리뷰 아코디언 HTML."""
    html_parts = []
    for label, color in [("긍정", "#10B981"), ("부정", "#EF4444"), ("중립", "#FBBF24")]:
        sub = df[df["감성"] == label].copy()
        if label == "긍정":
            sub = sub.sort_values("감성점수", ascending=False)
        elif label == "부정":
            sub = sub.sort_values("감성점수", ascending=True)
        else:
            sub = sub.head(5)

        samples = sub.head(5)
        if samples.empty:
            continue

        review_list = ""
        for _, row in samples.iterrows():
            text = str(row[text_col])[:200]
            score = row["감성점수"]
            keywords = row.get("매칭키워드", "")
            review_list += (
                f"<li style='margin-bottom:8px;padding:8px;background:#f9fafb;border-radius:6px;'>"
                f"<div style='font-size:0.9em;color:#374151;'>{text}</div>"
                f"<div style='font-size:0.8em;color:#6b7280;margin-top:4px;'>"
                f"점수: {score} | 키워드: {keywords}</div>"
                f"</li>"
            )

        html_parts.append(
            f"<details style='margin-bottom:8px;'>"
            f"<summary style='cursor:pointer;padding:10px;background:{color}22;"
            f"border-left:4px solid {color};border-radius:4px;font-weight:bold;'>"
            f"{label} 리뷰 ({len(sub):,}건 중 상위 {len(samples)}건)</summary>"
            f"<ul style='list-style:none;padding:8px;margin:0;'>{review_list}</ul>"
            f"</details>"
        )

    return "".join(html_parts) if html_parts else "<p>분석 가능한 리뷰가 없습니다.</p>"


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def run_sentiment(df: pd.DataFrame, job_id: str, chart_mode: str = "plotly") -> dict:
    """감성 분석 메인 함수.

    Returns:
        dict with keys: summary_html, charts, details_html, downloads
    """
    # 텍스트 컬럼 탐지
    text_col = detect_text_column(df)
    logger.info(f"[감성분석] 텍스트 컬럼: {text_col}, 행 수: {len(df)}")

    # 사전 로드
    lexicon = _load_lexicon()

    # 빈 데이터프레임 처리
    if df.empty or text_col not in df.columns:
        return {
            "summary_html": "<p>분석할 리뷰 데이터가 없습니다.</p>",
            "charts": [],
            "details_html": "<p>분석할 리뷰 데이터가 없습니다.</p>",
            "downloads": [],
        }

    # 각 리뷰 점수 및 키워드 매칭
    scores = []
    matched_keywords_list = []
    for text in df[text_col].fillna(""):
        score, matched = _score_review(str(text), lexicon)
        scores.append(score)
        matched_keywords_list.append(matched)

    df = df.copy()
    df["감성점수"] = scores
    df["감성"] = df["감성점수"].apply(_classify_sentiment)
    df["매칭키워드"] = [", ".join(kws) for kws in matched_keywords_list]

    # 통계 계산
    total = len(df)
    counts = df["감성"].value_counts().to_dict()
    avg_score = df["감성점수"].mean() if total > 0 else 0.0

    # 키워드 빈도 집계
    keyword_counter_pos = Counter()
    keyword_counter_neg = Counter()
    keyword_scores_for_wc = {}  # word -> lexicon score (for wordcloud coloring)

    for kws in matched_keywords_list:
        for w in kws:
            s = lexicon.get(w, 0)
            if s > 0:
                keyword_counter_pos[w] += 1
            elif s < 0:
                keyword_counter_neg[w] += 1
            # 워드클라우드용 점수 기록 (lexicon 기준)
            if w not in keyword_scores_for_wc:
                keyword_scores_for_wc[w] = s

    # 전체 키워드 빈도 집계 (워드클라우드 크기용)
    all_keyword_counter = Counter()
    for kws in matched_keywords_list:
        all_keyword_counter.update(kws)

    # 워드클라우드 색상용 점수 사전 (lexicon 원본 점수)
    wc_score_map = {word: lexicon.get(word, 0) for word in all_keyword_counter}

    # --- 차트 생성 ---
    charts = []

    # 1. 도넛 차트
    charts.append(_build_donut_chart(counts, chart_mode))

    # 2. 워드클라우드
    wc_chart = _build_wordcloud_chart(dict(all_keyword_counter), wc_score_map)
    if wc_chart:
        charts.append(wc_chart)

    # 3. 긍정/부정 TOP5 바 차트
    charts.append(_build_top_keywords_chart(keyword_counter_pos, keyword_counter_neg, chart_mode))

    # --- HTML 생성 ---
    summary_html = _build_summary_html(counts, total, avg_score)
    details_html = _build_details_html(df, text_col)

    # --- CSV 저장 ---
    result_dir = os.path.join(_BASE_DIR, "results", job_id)
    os.makedirs(result_dir, exist_ok=True)
    csv_filename = "감성_분석결과.csv"
    csv_path = os.path.join(result_dir, csv_filename)

    export_cols = [text_col, "감성점수", "감성", "매칭키워드"]
    # 원본 컬럼도 포함 (있으면)
    for col in df.columns:
        if col not in export_cols:
            export_cols.append(col)

    df[export_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"[감성분석] CSV 저장: {csv_path}")

    downloads = [{"filename": csv_filename, "label": "감성 분석 CSV"}]

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
