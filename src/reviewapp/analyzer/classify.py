"""리뷰 분류 모듈 - 키워드 기반 카테고리 분류."""
import os
import json
import pandas as pd

from reviewapp.app import detect_text_column
from reviewapp.analyzer.chart_utils import (
    plotly_donut,
    plotly_bar_h,
    plotly_to_json,
    plotly_save_png,
    bundle_zip,
)

# ---------------------------------------------------------------------------
# 카테고리 키워드 사전 (우선순위 순)
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("고객서비스", ["교환", "환불", "반품", "고객센터", "CS ", "문의", "답변이", "대응", "처리", "클레임", "상담", "연락", "전화", "응대", "사과", "보상", "환불받", "교환해", "AS "]),
    ("배송", ["배송", "택배", "포장", "도착", "파손", "박스", "운송", "배달", "수령", "에어캡", "뽁뽁이", "깨져서", "찌그러", "누락", "늦게 와", "빠르게 와", "하루만에", "당일", "안전하게", "꼼꼼하게"]),
    ("가격", ["가격", "가성비", "비싸", "비쌌", "저렴", "합리적", "가격대비", "가격 대비", "할인", "세일", "이벤트", "1+1", "1+2", "만원", "원에 ", "싸게", "착한 가격", "착한가격", "쿠폰", "적립", "본전", "가격이", "값", "비용", "저렴하", "갓성비"]),
    ("품질", ["품질", "퀄리티", "마감", "내구", "조잡", "불량", "하자", "튼튼", "견고", "약해", "새어", "누출", "변질", "오래 쓸", "오래 써", "수명", "고장", "망가", "뜯어", "갈라", "벗겨", "흠집", "기스", "찢어", "질이", "좋은 퀄", "싸구려", "고급", "제대로", "단단", "원단", "실밥", "올 나", "세탁", "비침", "내구성"]),
    ("디자인", ["디자인", "예쁘", "이쁘", "외관", "색상", "색깔", "모양", "패키지", "비주얼", "세련", "고급스러", "촌스러", "멋지", "심플", "화려", "귀엽", "인테리어", "감성적", "무드", "깔끔하", "예뻐", "이뻐", "컬러", "색감", "분위기"]),
    ("기능", ["기능", "성능", "효과", "사용", "편리", "편해", "불편", "사이즈", "크기", "무게", "가벼", "무거", "소음", "냄새", "향", "발림", "흡수", "지속", "세정", "세척", "작동", "설치", "조립", "충전", "배터리", "속도", "밝기", "온도", "보습", "탄력", "촉감", "착용감", "핏", "맛", "식감", "발향", "잔향", "지속력", "효능", "쓰기", "사용감", "착용", "입으면", "소리", "연결", "발열", "화질", "두께", "신선", "유통기한", "첨가물"]),
    ("만족도", ["좋아요", "좋아", "좋습니다", "좋네요", "좋고", "만족", "최고", "추천", "재구매", "또 살", "또 시킬", "굿", "good", "best", "대박", "짱", "강추", "별로", "실망", "후회", "안 좋", "최악", "비추", "그냥 그래", "아쉬", "괜찮", "무난", "쏘쏘", "감사", "잘 쓸", "잘 먹", "잘 쓰고", "잘 받았", "좋은", "좋았", "좋아서", "좋습", "좋네", "조아", "조아요", "잘쓰고", "잘쓰", "잘 쓰", "구매합", "구매했", "구매해", "또 주문", "또주문", "보통", "그럭저럭", "나쁘지", "넘 좋", "존좋", "쩐다", "ㄱㅊ", "개좋", "잘 먹", "또 올"]),
]

COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6B7280", "#94A3B8"]

UNCLASSIFIED = "미분류"


# ---------------------------------------------------------------------------
# 분류 함수
# ---------------------------------------------------------------------------

def _classify_text(text: str) -> str:
    """하나의 리뷰 텍스트를 우선순위 기반으로 분류. 매칭 없으면 미분류."""
    if not isinstance(text, str) or not text.strip():
        return UNCLASSIFIED
    for cat_name, keywords in CATEGORIES:
        for kw in keywords:
            if kw in text:
                return cat_name
    return UNCLASSIFIED


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def run_classification(df: pd.DataFrame, job_id: str, chart_mode: str = "plotly") -> dict:
    """키워드 매칭 기반 리뷰 분류를 실행하고 결과 dict를 반환."""

    # 1) 텍스트 컬럼 탐지
    text_col = detect_text_column(df)

    # 2) 빈값/NaN 처리 후 분류
    df["_review_text"] = df[text_col].fillna("").astype(str)
    df["카테고리"] = df["_review_text"].apply(_classify_text)

    # 3) 카테고리별 집계 (건수 내림차순, 미분류는 항상 마지막)
    counts = df["카테고리"].value_counts()
    cats_without_uncl = [(c, int(counts[c])) for c in counts.index if c != UNCLASSIFIED]
    cats_without_uncl.sort(key=lambda x: x[1], reverse=True)
    if UNCLASSIFIED in counts.index:
        cats_without_uncl.append((UNCLASSIFIED, int(counts[UNCLASSIFIED])))
    ordered_cats = [c for c, _ in cats_without_uncl]
    ordered_counts = [v for _, v in cats_without_uncl]
    total = sum(ordered_counts)

    # 4) 색상 매핑
    color_map = {}
    for i, (cat_name, _) in enumerate(CATEGORIES):
        color_map[cat_name] = COLORS[i % len(COLORS)]
    color_map[UNCLASSIFIED] = COLORS[len(CATEGORIES) % len(COLORS)]
    ordered_colors = [color_map[c] for c in ordered_cats]

    # -----------------------------------------------------------------------
    # 요약 HTML 테이블
    # -----------------------------------------------------------------------
    rows_html = ""
    for cat, cnt in zip(ordered_cats, ordered_counts):
        pct = cnt / total * 100 if total > 0 else 0
        color = color_map[cat]
        rows_html += (
            f"<tr>"
            f'<td><span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{color};margin-right:6px;"></span>{cat}</td>'
            f"<td >{cnt:,}건</td>"
            f"<td >{pct:.1f}%</td>"
            f"</tr>"
        )
    summary_html = (
        '<table class="table table-sm table-hover">'
        "<thead><tr><th>카테고리</th><th >건수</th>"
        "<th >비율</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f'<tfoot><tr style="font-weight:bold"><td>합계</td>'
        f'<td>{total:,}건</td>'
        f'<td>100.0%</td></tr></tfoot>'
        "</table>"
    )

    # -----------------------------------------------------------------------
    # 차트 생성 (Plotly 인터랙티브 + kaleido PNG 다운로드)
    # -----------------------------------------------------------------------
    charts = []
    result_dir = os.path.join(os.path.dirname(__file__), "..", "results", job_id)
    os.makedirs(result_dir, exist_ok=True)

    # -- 도넛 차트 --
    donut_fig = plotly_donut(
        labels=ordered_cats, values=ordered_counts,
        title="카테고리별 비율", colors=ordered_colors,
    )
    donut_json = json.loads(plotly_to_json(donut_fig))
    charts.append({"title": "카테고리별 비율", "plotly": donut_json})
    plotly_save_png(donut_fig, os.path.join(result_dir, "카테고리별_비율.png"))

    # -- 수평 바 차트 --
    bar_fig = plotly_bar_h(
        labels=ordered_cats, values=ordered_counts,
        title="카테고리별 리뷰 수", color=ordered_colors,
    )
    bar_json = json.loads(plotly_to_json(bar_fig))
    charts.append({"title": "카테고리별 리뷰 수", "plotly": bar_json})
    plotly_save_png(bar_fig, os.path.join(result_dir, "카테고리별_리뷰수.png"),
                    height=max(400, len(ordered_cats) * 50))

    # -----------------------------------------------------------------------
    # 상세 결과 - 카테고리별 대표 리뷰 아코디언
    # -----------------------------------------------------------------------
    accordion_items = ""
    for idx, cat in enumerate(ordered_cats):
        cat_df = df[df["카테고리"] == cat]
        sample_reviews = cat_df["_review_text"].head(5).tolist()
        reviews_html = ""
        for r in sample_reviews:
            display = r[:200] + "..." if len(r) > 200 else r
            reviews_html += f'<div class="acc-review">{display}</div>'
        if not sample_reviews:
            reviews_html = '<div class="acc-review" style="color:#999;">해당 카테고리 리뷰가 없습니다.</div>'

        cat_count = len(cat_df)
        open_cls = " open" if idx == 0 else ""
        accordion_items += (
            f'<div class="acc-item">'
            f'<button class="acc-btn{open_cls}" onclick="toggleAcc(this)">'
            f'{cat} ({cat_count:,}건)</button>'
            f'<div class="acc-body{open_cls}">{reviews_html}</div></div>'
        )

    details_html = accordion_items

    # -----------------------------------------------------------------------
    # CSV 저장
    # -----------------------------------------------------------------------
    csv_filename = "리뷰_분류결과.csv"
    csv_path = os.path.join(result_dir, csv_filename)

    # 임시 컬럼 제거 후 저장
    save_df = df.drop(columns=["_review_text"], errors="ignore")
    save_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------------
    # 반환
    # -----------------------------------------------------------------------
    # 차트 PNG를 ZIP으로 묶기
    chart_zip = bundle_zip(result_dir, "차트_이미지.zip", pattern="*.png")
    downloads = [
        {"filename": csv_filename, "label": "분류 결과 CSV"},
        {"filename": chart_zip, "label": "차트 이미지 ZIP"},
    ]

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
