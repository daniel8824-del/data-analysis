"""텍스트 마이닝 모듈 — TF-IDF, LDA, 키워드 네트워크, 워드클라우드, 감성 분석."""
import os
import json
import logging
import itertools

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import detect_text_column
from analyzer.chart_utils import (
    plotly_bar_h,
    plotly_network,
    plotly_donut,
    fig_to_base64,
    plotly_to_json,
    get_korean_font_path,
)

logger = logging.getLogger("review-analyzer")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# 불용어 로드
# ---------------------------------------------------------------------------

def _load_stopwords() -> set:
    path = os.path.join(DATA_DIR, "korean_stopwords.txt")
    words = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w)
    except FileNotFoundError:
        logger.warning("korean_stopwords.txt 파일을 찾을 수 없습니다.")
    return words

STOPWORDS = _load_stopwords()

# ---------------------------------------------------------------------------
# 감성 사전 로드
# ---------------------------------------------------------------------------

def _load_sentiment_lexicon() -> dict:
    path = os.path.join(DATA_DIR, "sentiment_lexicon.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("sentiment_lexicon.json 파일을 찾을 수 없습니다.")
        return {}

# ---------------------------------------------------------------------------
# 형태소 분석 (kiwipiepy)
# ---------------------------------------------------------------------------

_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize_texts(texts, pos_filter, stopwords):
    """텍스트 리스트를 형태소 분석하여 필터링된 토큰 리스트를 반환."""
    kiwi = _get_kiwi()
    result = []
    for text in texts:
        tokens = kiwi.tokenize(str(text))
        words = [
            t.form for t in tokens
            if t.tag in pos_filter
            and len(t.form) >= 2
            and t.form not in stopwords
        ]
        result.append(words)
    return result


# ---------------------------------------------------------------------------
# 유틸리티: 차트 출력 헬퍼
# ---------------------------------------------------------------------------

def _chart_entry(title, fig, chart_mode):
    """Plotly figure를 chart_mode에 따라 dict로 변환."""
    if chart_mode == "plotly":
        return {"title": title, "plotly": json.loads(plotly_to_json(fig))}
    else:
        # matplotlib static fallback — Plotly figure를 이미지로 변환
        try:
            img_bytes = fig.to_image(format="png", width=800, height=500)
            import base64
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            return {"title": title, "image": b64}
        except Exception:
            # kaleido 미설치 시 plotly json으로 폴백
            return {"title": title, "plotly": json.loads(plotly_to_json(fig))}


def _image_chart_entry(title, b64_image):
    """base64 이미지를 차트 dict로 감싸기."""
    return {"title": title, "image": b64_image}


# ===========================================================================
# 메인 함수
# ===========================================================================

def run_textmining(
    df: pd.DataFrame,
    job_id: str,
    chart_mode: str = "plotly",
    n_topics: int = 5,
    top_n: int = 20,
    pos_filter: list = None,
    analyses: list = None,
) -> dict:
    """텍스트 마이닝 파이프라인 실행."""

    if pos_filter is None:
        pos_filter = ["NNG", "NNP"]
    if analyses is None:
        analyses = ["tfidf", "lda", "network", "wordcloud", "sentiment"]

    # 결과 디렉토리
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 텍스트 컬럼 탐지
    text_col = detect_text_column(df)
    texts = df[text_col].fillna("").astype(str).tolist()
    total_docs = len(texts)

    # ----- 형태소 분석 -----
    logger.info(f"[텍스트마이닝] 형태소 분석 시작 — {total_docs}건")
    tokenized = tokenize_texts(texts, pos_filter, STOPWORDS)

    # 전체 단어 빈도
    all_words = [w for doc in tokenized for w in doc]
    word_freq = Counter(all_words)

    # 결과 컨테이너
    charts = []
    summary_parts = []
    details_parts = []
    downloads = []

    # 조인된 텍스트 (TF-IDF, LDA 공용)
    joined_texts = [" ".join(doc) for doc in tokenized]

    # ===================================================================
    # Step 2: TF-IDF 분석
    # ===================================================================
    if "tfidf" in analyses:
        try:
            logger.info("[텍스트마이닝] TF-IDF 분석")
            tfidf_vec = TfidfVectorizer(max_features=1000)
            tfidf_matrix = tfidf_vec.fit_transform(joined_texts)
            feature_names = tfidf_vec.get_feature_names_out()

            # 전체 문서 평균 TF-IDF 점수
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[::-1][:top_n]

            tfidf_keywords = [feature_names[i] for i in top_indices]
            tfidf_scores = [round(float(mean_tfidf[i]), 4) for i in top_indices]

            # 차트 (점수 낮은 순 → 위에서 아래로 높은 순)
            chart_labels = list(reversed(tfidf_keywords))
            chart_values = list(reversed(tfidf_scores))

            import plotly.graph_objects as go
            fig_tfidf = go.Figure(go.Bar(
                y=chart_labels, x=chart_values, orientation="h",
                text=[f"{v:.4f}" for v in chart_values],
                textposition="outside",
                marker_color="#4361ee",
            ))
            fig_tfidf.update_layout(
                title=dict(text=f"TF-IDF 키워드 TOP {top_n}", font=dict(size=16)),
                margin=dict(t=60, b=30, l=120, r=80),
                xaxis_title="TF-IDF 점수",
                yaxis=dict(tickfont=dict(size=11)),
            )
            charts.append(_chart_entry(f"TF-IDF 키워드 TOP {top_n}", fig_tfidf, chart_mode))

            # 요약 테이블
            rows = "".join(
                f"<tr><td>{i+1}</td><td>{kw}</td><td style='text-align:right'>{sc:.4f}</td></tr>"
                for i, (kw, sc) in enumerate(zip(tfidf_keywords[:10], tfidf_scores[:10]))
            )
            summary_parts.append(
                f"<h6>TF-IDF 상위 키워드</h6>"
                f'<table class="table table-sm table-hover">'
                f"<thead><tr><th>#</th><th>키워드</th><th style='text-align:right'>점수</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )

            # CSV 저장
            tfidf_df = pd.DataFrame({"키워드": tfidf_keywords, "TF-IDF점수": tfidf_scores})
            csv_name = "텍스트마이닝_키워드.csv"
            tfidf_df.to_csv(os.path.join(job_dir, csv_name), index=False, encoding="utf-8-sig")
            downloads.append({"filename": csv_name, "label": "TF-IDF 키워드 CSV"})

        except Exception as e:
            logger.error(f"[텍스트마이닝] TF-IDF 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">TF-IDF 분석 중 오류: {e}</div>')

    # ===================================================================
    # Step 3: LDA 토픽 모델링
    # ===================================================================
    if "lda" in analyses:
        try:
            logger.info("[텍스트마이닝] LDA 토픽 모델링")
            count_vec = CountVectorizer(max_features=1000)
            count_matrix = count_vec.fit_transform(joined_texts)
            feature_names_lda = count_vec.get_feature_names_out()

            actual_topics = min(n_topics, count_matrix.shape[0], len(feature_names_lda))
            if actual_topics < 2:
                raise ValueError("토픽 모델링에 충분한 데이터가 없습니다.")

            lda_model = LatentDirichletAllocation(
                n_components=actual_topics,
                max_iter=20,
                random_state=42,
                learning_method="online",
            )
            doc_topic_dist = lda_model.fit_transform(count_matrix)

            # 토픽별 상위 키워드
            topic_keywords = {}
            for topic_idx, topic_vec in enumerate(lda_model.components_):
                top_word_indices = topic_vec.argsort()[::-1][:10]
                words = [feature_names_lda[i] for i in top_word_indices]
                topic_keywords[f"토픽 {topic_idx+1}"] = words

            # 각 문서의 주요 토픽
            doc_topics = doc_topic_dist.argmax(axis=1)
            topic_counts = Counter(doc_topics)
            topic_labels = [f"토픽 {i+1}" for i in range(actual_topics)]
            topic_values = [int(topic_counts.get(i, 0)) for i in range(actual_topics)]

            # 도넛 차트: 토픽 점유율
            topic_colors = [
                "#4361ee", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
                "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
            ]
            fig_donut = plotly_donut(
                labels=topic_labels,
                values=topic_values,
                title="토픽 점유율",
                colors=topic_colors[:actual_topics],
            )
            charts.append(_chart_entry("토픽 점유율", fig_donut, chart_mode))

            # 상세 — 토픽 키워드 테이블
            topic_rows = ""
            for tname, twords in topic_keywords.items():
                cnt = topic_values[int(tname.split()[-1]) - 1]
                pct = cnt / total_docs * 100 if total_docs > 0 else 0
                topic_rows += (
                    f"<tr><td><strong>{tname}</strong></td>"
                    f"<td>{', '.join(twords)}</td>"
                    f"<td style='text-align:right'>{cnt:,}건 ({pct:.1f}%)</td></tr>"
                )
            details_parts.append(
                f"<h6>LDA 토픽별 키워드</h6>"
                f'<table class="table table-sm table-hover">'
                f"<thead><tr><th>토픽</th><th>주요 키워드</th><th style='text-align:right'>문서 수</th></tr></thead>"
                f"<tbody>{topic_rows}</tbody></table>"
            )

            # CSV 저장
            topic_csv_rows = []
            for tname, twords in topic_keywords.items():
                for rank, w in enumerate(twords, 1):
                    topic_csv_rows.append({"토픽": tname, "순위": rank, "키워드": w})
            topic_df = pd.DataFrame(topic_csv_rows)
            csv_name_topic = "텍스트마이닝_토픽.csv"
            topic_df.to_csv(os.path.join(job_dir, csv_name_topic), index=False, encoding="utf-8-sig")
            downloads.append({"filename": csv_name_topic, "label": "LDA 토픽 CSV"})

        except Exception as e:
            logger.error(f"[텍스트마이닝] LDA 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">LDA 분석 중 오류: {e}</div>')

    # ===================================================================
    # Step 4: 키워드 네트워크 분석
    # ===================================================================
    if "network" in analyses:
        try:
            logger.info("[텍스트마이닝] 키워드 네트워크 분석")

            # 대규모 데이터 샘플링
            net_tokenized = tokenized
            if len(tokenized) > 5000:
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(len(tokenized), 3000, replace=False)
                net_tokenized = [tokenized[i] for i in sample_idx]

            # 동시 출현 빈도 계산
            cooccurrence = defaultdict(int)
            for doc in net_tokenized:
                unique_words = list(set(doc))
                for w1, w2 in itertools.combinations(sorted(unique_words), 2):
                    cooccurrence[(w1, w2)] += 1

            # 상위 노드 선정: 전체 빈도 기준 top_n개
            top_words_set = set(w for w, _ in word_freq.most_common(top_n))

            # 그래프 구축: 최소 동시출현 2회 이상
            G = nx.Graph()
            for (w1, w2), count in cooccurrence.items():
                if count >= 2 and w1 in top_words_set and w2 in top_words_set:
                    G.add_edge(w1, w2, weight=count)

            # 고립 노드 제거 — 엣지가 있는 노드만 유지
            isolated = list(nx.isolates(G))
            G.remove_nodes_from(isolated)

            if len(G.nodes()) < 2:
                raise ValueError("네트워크를 구성할 수 있는 충분한 키워드 연결이 없습니다.")

            # 중심성 분석
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G, weight="weight")

            # 커뮤니티 탐지
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
            except Exception:
                # 폴백: 연결 요소 기반
                communities = list(nx.connected_components(G))

            community_colors_palette = [
                "#4361ee", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
                "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
                "#14B8A6", "#A855F7", "#F43F5E", "#0EA5E9", "#22C55E",
            ]
            node_community_map = {}
            for cidx, comm in enumerate(communities):
                for node in comm:
                    node_community_map[node] = cidx

            nodes = list(G.nodes())
            node_colors = [
                community_colors_palette[node_community_map.get(n, 0) % len(community_colors_palette)]
                for n in nodes
            ]

            # 노드 크기: degree centrality 기반 (최소 15, 최대 60)
            max_dc = max(degree_cent.values()) if degree_cent else 1
            node_sizes = [
                max(15, min(60, 15 + 45 * (degree_cent.get(n, 0) / max_dc)))
                for n in nodes
            ]

            # 레이아웃
            pos = nx.spring_layout(G, k=1.5, seed=42, weight="weight")

            fig_network = plotly_network(
                G, pos=pos,
                title="키워드 네트워크",
                node_colors=node_colors,
                node_sizes=node_sizes,
            )
            charts.append(_chart_entry("키워드 네트워크", fig_network, chart_mode))

            # 중심성 분석 TOP 10 바 차트
            cent_sorted = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            cent_labels = list(reversed([c[0] for c in cent_sorted]))
            cent_values = list(reversed([round(c[1], 4) for c in cent_sorted]))

            import plotly.graph_objects as go
            fig_cent = go.Figure(go.Bar(
                y=cent_labels, x=cent_values, orientation="h",
                text=[f"{v:.4f}" for v in cent_values],
                textposition="outside",
                marker_color="#8B5CF6",
            ))
            fig_cent.update_layout(
                title=dict(text="중심성 분석 TOP 10", font=dict(size=16)),
                margin=dict(t=60, b=30, l=120, r=80),
                xaxis_title="연결 중심성 (Degree Centrality)",
                yaxis=dict(tickfont=dict(size=11)),
            )
            charts.append(_chart_entry("중심성 분석 TOP 10", fig_cent, chart_mode))

            # 상세 — 중심성 테이블
            cent_rows = ""
            for rank, (word, dc) in enumerate(
                sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:15], 1
            ):
                bc = betweenness_cent.get(word, 0)
                comm_id = node_community_map.get(word, 0) + 1
                cent_rows += (
                    f"<tr><td>{rank}</td><td>{word}</td>"
                    f"<td style='text-align:right'>{dc:.4f}</td>"
                    f"<td style='text-align:right'>{bc:.4f}</td>"
                    f"<td style='text-align:center'>그룹 {comm_id}</td></tr>"
                )
            details_parts.append(
                f"<h6>키워드 중심성 분석</h6>"
                f'<table class="table table-sm table-hover">'
                f"<thead><tr><th>#</th><th>키워드</th>"
                f"<th style='text-align:right'>연결 중심성</th>"
                f"<th style='text-align:right'>매개 중심성</th>"
                f"<th style='text-align:center'>커뮤니티</th></tr></thead>"
                f"<tbody>{cent_rows}</tbody></table>"
            )

        except Exception as e:
            logger.error(f"[텍스트마이닝] 네트워크 분석 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">네트워크 분석 중 오류: {e}</div>')

    # ===================================================================
    # Step 5: 워드클라우드
    # ===================================================================
    if "wordcloud" in analyses:
        try:
            logger.info("[텍스트마이닝] 워드클라우드 생성")
            font_path = get_korean_font_path()
            if not font_path:
                raise FileNotFoundError("한글 폰트를 찾을 수 없습니다.")

            wc = WordCloud(
                font_path=font_path,
                width=800,
                height=500,
                background_color="white",
                max_words=100,
                colormap="viridis",
                prefer_horizontal=0.7,
            )
            wc.generate_from_frequencies(word_freq)

            fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.set_title("키워드 워드클라우드", fontsize=14)
            ax_wc.axis("off")
            plt.tight_layout()
            wc_b64 = fig_to_base64(fig_wc)
            charts.append(_image_chart_entry("키워드 워드클라우드", wc_b64))

        except Exception as e:
            logger.error(f"[텍스트마이닝] 워드클라우드 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">워드클라우드 생성 중 오류: {e}</div>')

    # ===================================================================
    # Step 6: 감성 분석 네트워크
    # ===================================================================
    if "sentiment" in analyses:
        try:
            logger.info("[텍스트마이닝] 감성 분석")
            sentiment_lex = _load_sentiment_lexicon()
            if not sentiment_lex:
                raise ValueError("감성 사전을 로드할 수 없습니다.")

            font_path = get_korean_font_path()
            if not font_path:
                raise FileNotFoundError("한글 폰트를 찾을 수 없습니다.")

            # ----- 감성 색상 워드클라우드 -----
            # 키워드별 감성 점수 결정
            def _sentiment_color_func(word, **kwargs):
                score = sentiment_lex.get(word, 0)
                if score > 0:
                    return "#22C55E"  # 긍정 — 녹색
                elif score < 0:
                    return "#EF4444"  # 부정 — 빨간색
                else:
                    return "#9CA3AF"  # 중립 — 회색

            wc_sent = WordCloud(
                font_path=font_path,
                width=800,
                height=500,
                background_color="white",
                max_words=100,
                prefer_horizontal=0.7,
            )
            wc_sent.generate_from_frequencies(word_freq)
            wc_sent.recolor(color_func=_sentiment_color_func)

            fig_swc, ax_swc = plt.subplots(figsize=(10, 6))
            ax_swc.imshow(wc_sent, interpolation="bilinear")
            ax_swc.set_title("긍부정 키워드 워드클라우드", fontsize=14)
            ax_swc.axis("off")

            # 범례 추가
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#22C55E", label="긍정"),
                Patch(facecolor="#EF4444", label="부정"),
                Patch(facecolor="#9CA3AF", label="중립"),
            ]
            ax_swc.legend(
                handles=legend_elements, loc="lower right",
                fontsize=10, framealpha=0.8,
            )
            plt.tight_layout()
            swc_b64 = fig_to_base64(fig_swc)
            charts.append(_image_chart_entry("긍부정 키워드 워드클라우드", swc_b64))

            # ----- 문서별 감성 분포 -----
            pos_count = 0
            neg_count = 0
            neu_count = 0

            for doc_tokens in tokenized:
                doc_score = 0
                for token in doc_tokens:
                    doc_score += sentiment_lex.get(token, 0)
                if doc_score > 0:
                    pos_count += 1
                elif doc_score < 0:
                    neg_count += 1
                else:
                    neu_count += 1

            sent_labels = ["긍정", "부정", "중립"]
            sent_values = [pos_count, neg_count, neu_count]
            sent_colors = ["#22C55E", "#EF4444", "#9CA3AF"]

            fig_sent_donut = plotly_donut(
                labels=sent_labels,
                values=sent_values,
                title="문서별 감성 분포",
                colors=sent_colors,
            )
            charts.append(_chart_entry("문서별 감성 분포", fig_sent_donut, chart_mode))

            # 요약에 감성 비율 추가
            sent_total = pos_count + neg_count + neu_count
            if sent_total > 0:
                summary_parts.append(
                    f"<h6>감성 분포</h6>"
                    f'<table class="table table-sm table-hover">'
                    f"<thead><tr><th>감성</th><th style='text-align:right'>건수</th>"
                    f"<th style='text-align:right'>비율</th></tr></thead>"
                    f"<tbody>"
                    f"<tr><td><span style='color:#22C55E'>&#9679;</span> 긍정</td>"
                    f"<td style='text-align:right'>{pos_count:,}건</td>"
                    f"<td style='text-align:right'>{pos_count/sent_total*100:.1f}%</td></tr>"
                    f"<tr><td><span style='color:#EF4444'>&#9679;</span> 부정</td>"
                    f"<td style='text-align:right'>{neg_count:,}건</td>"
                    f"<td style='text-align:right'>{neg_count/sent_total*100:.1f}%</td></tr>"
                    f"<tr><td><span style='color:#9CA3AF'>&#9679;</span> 중립</td>"
                    f"<td style='text-align:right'>{neu_count:,}건</td>"
                    f"<td style='text-align:right'>{neu_count/sent_total*100:.1f}%</td></tr>"
                    f"</tbody></table>"
                )

        except Exception as e:
            logger.error(f"[텍스트마이닝] 감성 분석 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">감성 분석 중 오류: {e}</div>')

    # ===================================================================
    # 결과 종합
    # ===================================================================

    # 전체 요약 헤더
    top10_words = word_freq.most_common(10)
    top10_html = ", ".join(f"<strong>{w}</strong>({c:,})" for w, c in top10_words)
    header_html = (
        f'<div class="mb-3">'
        f"<p>분석 문서: <strong>{total_docs:,}</strong>건 &nbsp;|&nbsp; "
        f"고유 키워드: <strong>{len(word_freq):,}</strong>개 &nbsp;|&nbsp; "
        f"분석 컬럼: <strong>{text_col}</strong></p>"
        f"<p>빈출 키워드 TOP 10: {top10_html}</p>"
        f"</div>"
    )

    summary_html = header_html + "\n".join(summary_parts)
    details_html = "\n".join(details_parts) if details_parts else ""

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
