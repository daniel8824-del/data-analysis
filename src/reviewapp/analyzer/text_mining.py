"""텍스트 마이닝 모듈 - TF-IDF, LDA, 키워드 네트워크, 워드클라우드, 감성 분석."""
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

from reviewapp.app import detect_text_column
from reviewapp.analyzer.chart_utils import (
    plotly_bar_h,
    plotly_network,
    plotly_donut,
    plotly_to_json,
    plotly_save_png,
    bundle_zip,
    fig_to_base64,
    get_korean_font_path,
    _apply_style,
    CHART_COLORS,
)

logger = logging.getLogger("review-analyzer")

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(os.path.expanduser("~"), ".review-analyzer")
RESULT_DIR = os.path.join(WORK_DIR, "results")
DATA_DIR = os.path.join(PKG_DIR, "data")

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

def _load_sentiment_lexicon(lang="ko") -> dict:
    if lang == "en":
        filename = "en_sentiment_lexicon.json"
    elif lang in ("zh", "ja"):
        filename = "en_sentiment_lexicon.json"  # fallback to English for now
    else:
        filename = "sentiment_lexicon.json"
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"{filename} 파일을 찾을 수 없습니다.")
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


def _detect_language(texts, sample=50):
    """텍스트 리스트에서 주 언어 감지 (ko/en/zh/ja)."""
    import re
    sample_text = " ".join(texts[:sample])
    ko_count = len(re.findall(r'[가-힣]', sample_text))
    en_count = len(re.findall(r'[a-zA-Z]', sample_text))
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', sample_text))
    ja_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', sample_text))

    counts = {'ko': ko_count, 'en': en_count, 'zh': zh_count, 'ja': ja_count}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else 'en'


# 영어 불용어
_EN_STOPWORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "not", "no", "nor", "as", "if", "than", "too", "very", "just", "about",
    "also", "so", "then", "there", "here", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "into", "over", "after", "before", "between", "under", "again", "further",
    "once", "during", "while", "up", "down", "out", "off", "through",
])


def tokenize_texts(texts, pos_filter, stopwords):
    """텍스트 리스트를 형태소 분석하여 (토큰 리스트, 언어코드) 반환."""
    lang = _detect_language(texts)

    if lang == "ko":
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
        return result, lang
    elif lang == "ja":
        try:
            import fugashi
            tagger = fugashi.Tagger()
            result = []
            for text in texts:
                words = [word.surface for word in tagger(str(text))
                         if len(word.surface) >= 2 and word.surface not in stopwords]
                result.append(words)
            return result, lang
        except ImportError:
            # Fallback: character-based tokenization
            import re
            result = []
            for text in texts:
                # Extract kanji+hiragana sequences of 2+ chars
                words = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}', str(text))
                words = [w for w in words if w not in stopwords]
                result.append(words)
            return result, lang
    elif lang == "zh":
        try:
            import jieba
            result = []
            for text in texts:
                words = [w for w in jieba.cut(str(text)) if len(w) >= 2 and w not in stopwords]
                result.append(words)
            return result, lang
        except ImportError:
            # Fallback: extract 2+ char sequences
            import re
            result = []
            for text in texts:
                words = re.findall(r'[\u4e00-\u9fff]{2,}', str(text))
                words = [w for w in words if w not in stopwords]
                result.append(words)
            return result, lang
    else:
        # 영어: 단순 토큰화 + lemmatization + 불용어 제거
        import re
        from nltk.stem import WordNetLemmatizer
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        lemmatizer = WordNetLemmatizer()
        result = []
        for text in texts:
            words = re.findall(r'[a-zA-Z]{3,}', str(text).lower())
            words = [lemmatizer.lemmatize(w) for w in words]
            words = [w for w in words if w not in _EN_STOPWORDS and w not in stopwords]
            result.append(words)
        return result, lang


# ---------------------------------------------------------------------------
# 유틸리티: 차트 출력 헬퍼
# ---------------------------------------------------------------------------

def _chart_entry(title, fig):
    """Plotly figure를 dict로 변환."""
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
    logger.info(f"[텍스트마이닝] 형태소 분석 시작 - {total_docs}건")
    tokenized, detected_lang = tokenize_texts(texts, pos_filter, STOPWORDS)

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
                marker_color=CHART_COLORS[0],
                hovertext=[f"{kw}: {v:.4f}" for kw, v in zip(chart_labels, chart_values)],
                hoverinfo="text",
            ))
            _apply_style(fig_tfidf,
                title=dict(text=f"TF-IDF 키워드 TOP {top_n}", font=dict(size=16)),
                margin=dict(t=60, b=30, l=120, r=80),
                xaxis_title="TF-IDF 점수",
                yaxis=dict(tickfont=dict(size=11)),
            )
            charts.append(_chart_entry(f"TF-IDF 키워드 TOP {top_n}", fig_tfidf))
            plotly_save_png(fig_tfidf, os.path.join(job_dir, "tfidf_keywords.png"))

            # 요약 테이블
            rows = "".join(
                f"<tr><td>{i+1}</td><td>{kw}</td><td>{sc:.4f}</td></tr>"
                for i, (kw, sc) in enumerate(zip(tfidf_keywords[:10], tfidf_scores[:10]))
            )
            summary_parts.append(
                '<div style="height:14px"></div><h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">TF-IDF 상위 키워드</h3>'
                f"<table>"
                f"<thead><tr><th>#</th><th>키워드</th><th>점수</th></tr></thead>"
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
    # Step 2.5: N-gram Analysis (Bigram / Trigram)
    # ===================================================================
    try:
        if len(all_words) >= 10:
            logger.info("[텍스트마이닝] N-gram 분석")
            from collections import Counter as _Counter
            bigram_counter = _Counter()
            trigram_counter = _Counter()
            for doc in tokenized:
                for i in range(len(doc) - 1):
                    bigram_counter[(doc[i], doc[i+1])] += 1
                for i in range(len(doc) - 2):
                    trigram_counter[(doc[i], doc[i+1], doc[i+2])] += 1

            if bigram_counter:
                # 차트: 상위 15 바이그램 가로 바 차트
                top15_bigrams = bigram_counter.most_common(15)
                bg_labels = [f"{w1} + {w2}" for (w1, w2), _ in top15_bigrams]
                bg_values = [cnt for _, cnt in top15_bigrams]

                chart_bg_labels = list(reversed(bg_labels))
                chart_bg_values = list(reversed(bg_values))

                import plotly.graph_objects as go
                fig_ngram = go.Figure(go.Bar(
                    y=chart_bg_labels, x=chart_bg_values, orientation="h",
                    text=[str(v) for v in chart_bg_values],
                    textposition="outside",
                    marker_color=CHART_COLORS[2] if len(CHART_COLORS) > 2 else CHART_COLORS[0],
                    hovertext=[f"{kw}: {v}회" for kw, v in zip(chart_bg_labels, chart_bg_values)],
                    hoverinfo="text",
                ))
                _apply_style(fig_ngram,
                    title=dict(text="바이그램 TOP 15", font=dict(size=16)),
                    margin=dict(t=60, b=30, l=160, r=80),
                    xaxis_title="빈도",
                    yaxis=dict(tickfont=dict(size=11)),
                )
                charts.append(_chart_entry("바이그램 TOP 15", fig_ngram))
                plotly_save_png(fig_ngram, os.path.join(job_dir, "bigram_top15.png"))

                # 요약 테이블: 상위 10 바이그램 + 상위 10 트라이그램
                top10_bigrams = bigram_counter.most_common(10)
                top10_trigrams = trigram_counter.most_common(10)

                ngram_rows = ""
                for i in range(10):
                    bg_cell = ""
                    bg_freq = ""
                    tg_cell = ""
                    tg_freq = ""
                    if i < len(top10_bigrams):
                        (w1, w2), cnt = top10_bigrams[i]
                        bg_cell = f"{w1} + {w2}"
                        bg_freq = f"{cnt:,}"
                    if i < len(top10_trigrams):
                        (w1, w2, w3), cnt = top10_trigrams[i]
                        tg_cell = f"{w1} + {w2} + {w3}"
                        tg_freq = f"{cnt:,}"
                    ngram_rows += f"<tr><td>{i+1}</td><td>{bg_cell}</td><td>{bg_freq}</td><td>{tg_cell}</td><td>{tg_freq}</td></tr>"

                summary_parts.append(
                    '<div style="height:14px"></div>'
                    '<h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">N-gram 분석</h3>'
                    "<table>"
                    "<thead><tr><th>#</th><th>바이그램</th><th>빈도</th><th>트라이그램</th><th>빈도</th></tr></thead>"
                    f"<tbody>{ngram_rows}</tbody></table>"
                )

                # CSV 저장
                ngram_csv_rows = []
                for (w1, w2), cnt in bigram_counter.most_common(50):
                    ngram_csv_rows.append({"유형": "바이그램", "N-gram": f"{w1} + {w2}", "빈도": cnt})
                for (w1, w2, w3), cnt in trigram_counter.most_common(50):
                    ngram_csv_rows.append({"유형": "트라이그램", "N-gram": f"{w1} + {w2} + {w3}", "빈도": cnt})
                ngram_df = pd.DataFrame(ngram_csv_rows)
                csv_name_ngram = "텍스트마이닝_N-gram.csv"
                ngram_df.to_csv(os.path.join(job_dir, csv_name_ngram), index=False, encoding="utf-8-sig")
                downloads.append({"filename": csv_name_ngram, "label": "N-gram CSV"})

    except Exception as e:
        logger.error(f"[텍스트마이닝] N-gram 분석 오류: {e}")
        summary_parts.append(f'<div class="alert alert-warning">N-gram 분석 중 오류: {e}</div>')

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

            # 도넛 차트: 토픽 점유율 (도넛 안은 짧게, 범례에 키워드)
            legend_labels = [f"{tl}: {', '.join(topic_keywords[tl][:3])}" for tl in topic_labels]
            hover_texts = [f"{tl}<br>{', '.join(topic_keywords[tl][:5])}<br>{topic_values[i]:,}건" for i, tl in enumerate(topic_labels)]
            total_tv = sum(topic_values) or 1
            inside_text = [f"{tl}<br>{v/total_tv*100:.1f}%" if v/total_tv >= 0.05 else "" for tl, v in zip(topic_labels, topic_values)]
            fig_donut = plotly_donut(
                labels=legend_labels,
                values=topic_values,
                title="토픽 점유율",
                colors=CHART_COLORS[:actual_topics],
            )
            fig_donut.update_traces(text=inside_text, hovertext=hover_texts, hoverinfo="text")
            charts.append(_chart_entry("토픽 점유율", fig_donut))
            plotly_save_png(fig_donut, os.path.join(job_dir, "topic_distribution.png"))

            # 상세 - 토픽 키워드 테이블
            topic_rows = ""
            for tname, twords in topic_keywords.items():
                cnt = topic_values[int(tname.split()[-1]) - 1]
                pct = cnt / total_docs * 100 if total_docs > 0 else 0
                topic_rows += (
                    f"<tr><td><strong>{tname}</strong></td>"
                    f"<td>{', '.join(twords)}</td>"
                    f"<td>{cnt:,}건 ({pct:.1f}%)</td></tr>"
                )
            details_parts.append(
                '<div style="height:14px"></div><h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">LDA 토픽별 키워드</h3>'
                f"<table>"
                f"<thead><tr><th>토픽</th><th>주요 키워드</th><th>문서 수</th></tr></thead>"
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

            # 고립 노드 제거 - 엣지가 있는 노드만 유지
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

            community_colors_palette = CHART_COLORS + [
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
            charts.append(_chart_entry("키워드 네트워크", fig_network))
            plotly_save_png(fig_network, os.path.join(job_dir, "keyword_network.png"))

            # 중심성 분석 TOP 10 바 차트
            cent_sorted = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
            cent_labels = list(reversed([c[0] for c in cent_sorted]))
            cent_values = list(reversed([round(c[1], 4) for c in cent_sorted]))

            import plotly.graph_objects as go
            fig_cent = go.Figure(go.Bar(
                y=cent_labels, x=cent_values, orientation="h",
                text=[f"{v:.4f}" for v in cent_values],
                textposition="outside",
                marker_color=CHART_COLORS[4] if len(CHART_COLORS) > 4 else CHART_COLORS[0],
                hovertext=[f"{kw}: {v:.4f}" for kw, v in zip(cent_labels, cent_values)],
                hoverinfo="text",
            ))
            _apply_style(fig_cent,
                title=dict(text="중심성 분석 TOP 10", font=dict(size=16)),
                margin=dict(t=60, b=30, l=120, r=80),
                xaxis_title="연결 중심성 (Degree Centrality)",
                yaxis=dict(tickfont=dict(size=11)),
            )
            charts.append(_chart_entry("중심성 분석 TOP 10", fig_cent))
            plotly_save_png(fig_cent, os.path.join(job_dir, "centrality_top10.png"))

            # 상세 - 중심성 테이블
            cent_rows = ""
            for rank, (word, dc) in enumerate(
                sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:15], 1
            ):
                bc = betweenness_cent.get(word, 0)
                comm_id = node_community_map.get(word, 0) + 1
                cent_rows += (
                    f"<tr><td>{rank}</td><td>{word}</td>"
                    f"<td>{dc:.4f}</td>"
                    f"<td>{bc:.4f}</td>"
                    f"<td>그룹 {comm_id}</td></tr>"
                )
            details_parts.append(
                '<div style="height:14px"></div><h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">키워드 중심성 분석</h3>'
                f"<table>"
                f"<thead><tr><th>#</th><th>키워드</th>"
                f"<th>연결 중심성</th>"
                f"<th>매개 중심성</th>"
                f"<th>커뮤니티</th></tr></thead>"
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

            # Save PNG directly via matplotlib
            fig_wc.savefig(os.path.join(job_dir, "wordcloud.png"), format="png", dpi=150, bbox_inches="tight")

            wc_b64 = fig_to_base64(fig_wc)
            charts.append(_image_chart_entry("키워드 워드클라우드", wc_b64))

        except Exception as e:
            logger.error(f"[텍스트마이닝] 워드클라우드 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">워드클라우드 생성 중 오류: {e}</div>')

    # ===================================================================
    # Step 6: 감성 분석
    # ===================================================================
    if "sentiment" in analyses:
        try:
            logger.info("[텍스트마이닝] 감성 분석")
            sentiment_lex = _load_sentiment_lexicon(detected_lang)
            if not sentiment_lex:
                raise ValueError("감성 사전을 로드할 수 없습니다.")

            font_path = get_korean_font_path()

            # ----- 감성 색상 워드클라우드 -----
            def _sentiment_color_func(word, **kwargs):
                score = sentiment_lex.get(word, 0)
                if score > 0:
                    return "#22C55E"  # 긍정 - 녹색
                elif score < 0:
                    return "#EF4444"  # 부정 - 빨간색
                else:
                    return "#9CA3AF"  # 중립 - 회색

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

            # Save PNG directly via matplotlib
            fig_swc.savefig(os.path.join(job_dir, "sentiment_wordcloud.png"), format="png", dpi=150, bbox_inches="tight")

            swc_b64 = fig_to_base64(fig_swc)
            charts.append(_image_chart_entry("긍부정 키워드 워드클라우드", swc_b64))

            # ----- 문서별 감성 분포 -----
            pos_count = 0
            neg_count = 0
            neu_count = 0
            pos_examples = []
            neg_examples = []

            for i, doc_tokens in enumerate(tokenized):
                doc_score = 0
                for token in doc_tokens:
                    doc_score += sentiment_lex.get(token, 0)
                if doc_score > 0:
                    pos_count += 1
                    if len(pos_examples) < 3:
                        pos_examples.append(str(texts[i])[:120])
                elif doc_score < 0:
                    neg_count += 1
                    if len(neg_examples) < 3:
                        neg_examples.append(str(texts[i])[:120])
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
            charts.append(_chart_entry("문서별 감성 분포", fig_sent_donut))
            plotly_save_png(fig_sent_donut, os.path.join(job_dir, "sentiment_distribution.png"))

            # 요약에 감성 비율 + 예시 추가
            sent_total = pos_count + neg_count + neu_count
            if sent_total > 0:
                ex_html = ""

                summary_parts.append(
                    '<div style="height:14px"></div>'
                    '<h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">감성 분포</h3>'
                    "<table>"
                    "<thead><tr><th>감성</th><th>건수</th>"
                    "<th>비율</th></tr></thead>"
                    "<tbody>"
                    f"<tr><td><span style='color:#22C55E'>&#9679;</span> 긍정</td>"
                    f"<td>{pos_count:,}건</td>"
                    f"<td>{pos_count/sent_total*100:.1f}%</td></tr>"
                    f"<tr><td><span style='color:#EF4444'>&#9679;</span> 부정</td>"
                    f"<td>{neg_count:,}건</td>"
                    f"<td>{neg_count/sent_total*100:.1f}%</td></tr>"
                    f"<tr><td><span style='color:#9CA3AF'>&#9679;</span> 중립</td>"
                    f"<td>{neu_count:,}건</td>"
                    f"<td>{neu_count/sent_total*100:.1f}%</td></tr>"
                    "</tbody></table>"
                    f"{ex_html}"
                )

        except Exception as e:
            logger.error(f"[텍스트마이닝] 감성 분석 오류: {e}")
            summary_parts.append(f'<div class="alert alert-warning">감성 분석 중 오류: {e}</div>')

    # ===================================================================
    # 결과 종합
    # ===================================================================

    # 차트 이미지 ZIP 번들
    chart_zip = bundle_zip(job_dir, "차트_이미지.zip", pattern="*.png")
    downloads.append({"filename": chart_zip, "label": "차트 이미지 ZIP"})

    # 전체 요약 헤더 (테이블 형태)
    top10_words = word_freq.most_common(10)
    top10_rows = "".join(
        f"<tr><td>{i+1}</td><td>{w}</td><td>{c:,}</td></tr>"
        for i, (w, c) in enumerate(top10_words)
    )
    header_html = (
        '<h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">분석 개요</h3>'
        '<table>'
        '<thead><tr><th>항목</th><th>값</th></tr></thead>'
        '<tbody>'
        f'<tr><td>분석 문서</td><td>{total_docs:,}건</td></tr>'
        f'<tr><td>고유 키워드</td><td>{len(word_freq):,}개</td></tr>'
        f'<tr><td>분석 컬럼</td><td>{text_col}</td></tr>'
        f'<tr><td>감지 언어</td><td>{detected_lang.upper()}</td></tr>'
        '</tbody></table>'
        '<div style="height:14px"></div>'
        '<h3 style="font-size:14px;font-weight:600;color:#01696F;margin-bottom:8px;">빈출 키워드 TOP 10</h3>'
        '<table>'
        '<thead><tr><th>#</th><th>키워드</th><th>빈도</th></tr></thead>'
        f'<tbody>{top10_rows}</tbody></table>'
    )

    summary_html = header_html + "\n".join(summary_parts)
    details_html = "\n".join(details_parts) if details_parts else ""

    return {
        "summary_html": summary_html,
        "charts": charts,
        "details_html": details_html,
        "downloads": downloads,
    }
