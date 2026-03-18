# 이커머스 리뷰/데이터 분석 웹앱 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이커머스 클래스 Chapter 4 수업 순서에 맞춘 5탭 데이터 분석 웹앱 (Railway/Render 배포)

**Architecture:** FastAPI + Jinja2 서버 사이드 렌더링. 각 탭은 독립 분석 모듈로 분리. 파일 업로드 → 서버 분석 → Plotly/matplotlib 차트 + 결과 테이블 → CSV/PNG 다운로드. 한글 형태소 분석은 kiwipiepy (Java 불필요).

**Tech Stack:** FastAPI, Jinja2, Bootstrap 5, Plotly.js, matplotlib, kiwipiepy, scikit-learn, networkx, wordcloud, pandas, openpyxl

---

## 수업 순서 → 탭 매핑

```
수업 목차                         웹앱 탭
──────────────────────────────────────────
2. 리뷰 분류                  → 탭1: 리뷰 분류
3. 감성 분석                  → 탭2: 감성 분석
4. 탐색적 데이터 분석          → 탭3: EDA
5. 커머스 데이터 분석          → (탭3 공용)
6. TCP 분석                   → 탭4: TCP/RFM
7. 비즈니스 인사이트           → (각 탭 인사이트 포함)
8~12. 텍스트 마이닝 전체       → 탭5: 텍스트 마이닝
```

## File Structure

```
review-analyzer/
├── app.py                          # FastAPI 메인 앱 (라우팅, 파일 업로드)
├── analyzer/
│   ├── __init__.py
│   ├── classify.py                 # 탭1: 리뷰 분류 (키워드 매칭)
│   ├── sentiment.py                # 탭2: 감성 분석 (감성 사전 점수제)
│   ├── eda.py                      # 탭3: 탐색적 데이터 분석
│   ├── tcp_rfm.py                  # 탭4: TCP/RFM 분석
│   ├── text_mining.py              # 탭5: 텍스트 마이닝 (TF-IDF, LDA, 네트워크, 워드클라우드)
│   └── chart_utils.py              # 공용: Plotly/matplotlib 차트 생성 유틸
├── data/
│   ├── sentiment_lexicon.json      # 감성 사전 (강한긍정+2 ~ 강한부정-2)
│   └── korean_stopwords.txt        # 한국어 불용어
├── templates/
│   ├── base.html                   # 공용 레이아웃 (Bootstrap 5 + 탭 네비게이션)
│   ├── index.html                  # 메인 페이지 (탭 선택 + 파일 업로드)
│   └── partials/
│       ├── tab_classify.html       # 탭1 결과 영역
│       ├── tab_sentiment.html      # 탭2 결과 영역
│       ├── tab_eda.html            # 탭3 결과 영역
│       ├── tab_tcp.html            # 탭4 결과 영역
│       └── tab_textmining.html     # 탭5 결과 영역
├── static/
│   ├── css/style.css               # 커스텀 스타일
│   └── js/app.js                   # 파일 업로드, 탭 전환, 차트 토글
├── fonts/
│   └── NanumGothic.ttf             # 한글 폰트 (워드클라우드용)
├── uploads/                        # 임시 업로드 폴더 (자동 생성)
├── results/                        # 분석 결과 폴더 (자동 생성)
├── requirements.txt
├── Dockerfile
├── railway.toml
└── render.yaml
```

---

## Chunk 1: 프로젝트 초기 설정 + 공용 UI

### Task 1: 프로젝트 스캐폴딩

**Files:**
- Create: `review-analyzer/requirements.txt`
- Create: `review-analyzer/app.py`
- Create: `review-analyzer/analyzer/__init__.py`
- Create: `review-analyzer/analyzer/chart_utils.py`

- [ ] **Step 1: requirements.txt 작성**

```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9
jinja2==3.1.4
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
seaborn>=0.13
plotly>=5.18
wordcloud>=1.9
scikit-learn>=1.3
networkx>=3.1
kiwipiepy>=0.18
python-dotenv>=1.0
gunicorn>=21.2
kaleido>=0.2.1
```

- [ ] **Step 2: app.py 기본 구조 작성**

FastAPI 앱 생성, CORS, 정적 파일, 템플릿, 파일 업로드 라우트.
5개 분석 엔드포인트: POST `/api/classify`, `/api/sentiment`, `/api/eda`, `/api/tcp`, `/api/textmining`
각 엔드포인트는 업로드 파일 + 옵션 파라미터 받아서 분석 모듈 호출 → JSON 결과 반환.
결과 다운로드: GET `/api/download/{job_id}/{filename}`

- [ ] **Step 3: chart_utils.py 작성**

공용 함수:
- `create_plotly_chart(data, chart_type, title, ...)` → Plotly JSON
- `create_matplotlib_chart(data, chart_type, title, ...)` → PNG base64
- `get_korean_font_path()` → 한글 폰트 경로 탐색
- `save_chart(fig, path, format)` → 파일 저장

차트 타입: donut, bar_h, histogram, heatmap, network, wordcloud, line, scatter

- [ ] **Step 4: analyzer/__init__.py**

빈 파일 또는 공용 상수 (CATEGORIES, SENTIMENT_LEXICON 등)

- [ ] **Step 5: 커밋**

```bash
git init && git add -A && git commit -m "feat: project scaffolding"
```

### Task 2: 공용 UI 템플릿

**Files:**
- Create: `templates/base.html`
- Create: `templates/index.html`
- Create: `static/css/style.css`
- Create: `static/js/app.js`

- [ ] **Step 1: base.html — Bootstrap 5 레이아웃**

```html
<!-- Bootstrap 5.3 CDK, Plotly.js CDN, Font Awesome 6, Noto Sans KR -->
<!-- 헤더: 앱 타이틀 -->
<!-- 메인: 5개 탭 (Bootstrap nav-tabs) — 수업 순서 -->
<!-- 각 탭: 파일 업로드 영역 + 옵션 + 분석 버튼 + 결과 영역 -->
<!-- 푸터 -->
```

탭 순서 (수업 순서 그대로):
1. 리뷰 분류
2. 감성 분석
3. EDA
4. TCP/RFM
5. 텍스트 마이닝

- [ ] **Step 2: index.html — 각 탭 콘텐츠**

각 탭에 공통 구조:
- 파일 업로드 (드래그앤드롭 + 파일선택)
- 옵션 영역 (탭별 다름)
- 차트 모드 토글: `인터랙티브(Plotly)` / `정적(이미지)` 라디오버튼
- 분석 시작 버튼
- 로딩 스피너
- 결과 영역 (차트 + 테이블 + 다운로드)

- [ ] **Step 3: style.css — 커스텀 스타일**

Mining 프로젝트 스타일 참고: --primary: #4361ee, 그라데이션 헤더, 카드 그림자, 반응형

- [ ] **Step 4: app.js — 파일 업로드 + 탭 로직**

- 파일 업로드 핸들러 (FormData → fetch POST)
- 차트 모드 토글 (plotly/static)
- 분석 결과 렌더링 (Plotly JSON → plotly.newPlot / 이미지 → img src)
- CSV/PNG 다운로드 버튼 핸들러
- 로딩 스피너 on/off

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "feat: base UI with 5 tabs"
```

---

## Chunk 2: 탭1 리뷰 분류 + 탭2 감성 분석

### Task 3: 탭1 — 리뷰 분류 모듈

**Files:**
- Create: `analyzer/classify.py`
- Create: `templates/partials/tab_classify.html`

- [ ] **Step 1: classify.py 작성**

수업 프롬프트 코드를 모듈화:
- `classify_reviews(df, review_col=None)` → DataFrame에 "카테고리" 컬럼 추가
- `get_summary(df)` → 카테고리별 건수/비율 dict
- `get_representative_reviews(df, n=3)` → 카테고리별 대표 리뷰
- `create_charts(df, chart_mode="plotly")` → 도넛 + 막대 차트

키워드 사전: CATEGORIES (수업 프롬프트와 동일한 7개 카테고리)
자동 컬럼 탐지: 텍스트 길이 + 이름 매칭

- [ ] **Step 2: app.py에 /api/classify 엔드포인트 추가**

```python
@app.post("/api/classify")
async def analyze_classify(file: UploadFile, chart_mode: str = Form("plotly")):
    # 파일 저장 → DataFrame 로드 → classify_reviews() → 결과 JSON 반환
```

- [ ] **Step 3: tab_classify.html 결과 영역**

- 요약 테이블 (카테고리 | 건수 | 비율)
- 차트 2개 (도넛, 막대) — chart_mode에 따라 Plotly/이미지
- 카테고리별 대표 리뷰 아코디언
- CSV 다운로드 버튼

- [ ] **Step 4: 커밋**

```bash
git add -A && git commit -m "feat: tab1 review classification"
```

### Task 4: 탭2 — 감성 분석 모듈

**Files:**
- Create: `analyzer/sentiment.py`
- Create: `data/sentiment_lexicon.json`
- Create: `templates/partials/tab_sentiment.html`

- [ ] **Step 1: sentiment_lexicon.json 작성**

수업 프롬프트의 SENTIMENT_LEXICON을 JSON으로:
```json
{"최고": 2, "완벽": 2, ..., "최악": -2, "실망": -2, ...}
```

- [ ] **Step 2: sentiment.py 작성**

- `analyze_sentiment(df, review_col=None)` → DataFrame에 "감성점수", "감성" 컬럼 추가
- `get_summary(df)` → 긍정/부정/중립 건수/비율
- `get_top_keywords(df, sentiment, n=5)` → 긍정/부정 키워드 TOP N
- `get_representative_reviews(df, n=5)` → 감성별 대표 리뷰
- `create_charts(df, chart_mode="plotly")` → 도넛 + 워드클라우드 + TOP5 막대
- `create_wordcloud(df)` → 감성 키워드 워드클라우드 (초록=긍정, 빨강=부정)

- [ ] **Step 3: app.py에 /api/sentiment 엔드포인트 추가**

- [ ] **Step 4: tab_sentiment.html 결과 영역**

- 요약 테이블 + 평균 점수
- 차트 3개 (도넛, 워드클라우드, 긍정/부정 TOP5)
- 감성별 대표 리뷰 아코디언
- CSV 다운로드

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "feat: tab2 sentiment analysis"
```

---

## Chunk 3: 탭3 EDA + 탭4 TCP/RFM

### Task 5: 탭3 — EDA 모듈

**Files:**
- Create: `analyzer/eda.py`
- Create: `templates/partials/tab_eda.html`

- [ ] **Step 1: eda.py 작성**

자동 EDA 분석기:
- `auto_eda(df)` → 데이터 구조, 결측치, 기초 통계량 자동 파악
- `detect_column_types(df)` → 수치/범주/날짜/텍스트 자동 분류
- `generate_charts(df, chart_mode="plotly")`:
  - 수치 컬럼: 히스토그램 + 박스플롯
  - 범주 컬럼: 막대 차트 (TOP 15)
  - 날짜 컬럼: 시계열 라인 차트
  - 상관관계: 히트맵
  - 이상치: 박스플롯
- `get_insights(df)` → 자동 인사이트 텍스트 (선택: OpenRouter API)

옵션 UI:
- 분석할 컬럼 선택 (멀티셀렉트)
- 차트 모드 (plotly/static)

- [ ] **Step 2: app.py에 /api/eda 엔드포인트 추가**

- [ ] **Step 3: tab_eda.html**

- 데이터 구조 요약 테이블 (컬럼명, 타입, 결측치, 고유값 수)
- 기초 통계량 테이블
- 자동 생성 차트들 (Plotly/이미지)
- CSV 다운로드 (전처리된 데이터)

- [ ] **Step 4: 커밋**

```bash
git add -A && git commit -m "feat: tab3 EDA"
```

### Task 6: 탭4 — TCP/RFM 분석 모듈

**Files:**
- Create: `analyzer/tcp_rfm.py`
- Create: `templates/partials/tab_tcp.html`

- [ ] **Step 1: tcp_rfm.py 작성**

- `detect_commerce_columns(df)` → 날짜/고객ID/상품/수량/단가 컬럼 자동 탐지
- `time_analysis(df, chart_mode)` → 일/주/월별 매출 추이 (라인 차트)
- `product_analysis(df, chart_mode)` → 상품별/카테고리별 매출 (막대 차트)
- `rfm_analysis(df, chart_mode)`:
  - R/F/M 계산
  - 고객 세그먼트 분류 (Champions, Loyal, At Risk 등)
  - 세그먼트별 분포 (도넛)
  - RFM 스캐터 (3D Plotly)
- `generate_insights(results)` → 인사이트 텍스트

옵션 UI:
- 컬럼 매핑 (날짜, 고객ID, 상품명, 수량, 단가 — 자동 탐지 + 수동 선택)
- 분석 항목 체크박스 (시간분석, 상품분석, RFM)

- [ ] **Step 2: app.py에 /api/tcp 엔드포인트 추가**

- [ ] **Step 3: tab_tcp.html**

- 컬럼 매핑 UI (자동 탐지 결과 표시 + 드롭다운 수정)
- 시간별 매출 차트
- 상품별 매출 차트
- RFM 세그먼트 테이블 + 차트
- CSV 다운로드 (RFM 테이블)

- [ ] **Step 4: 커밋**

```bash
git add -A && git commit -m "feat: tab4 TCP/RFM analysis"
```

---

## Chunk 4: 탭5 텍스트 마이닝 + 배포

### Task 7: 탭5 — 텍스트 마이닝 모듈

**Files:**
- Create: `analyzer/text_mining.py`
- Create: `data/korean_stopwords.txt`
- Create: `templates/partials/tab_textmining.html`

- [ ] **Step 1: korean_stopwords.txt 작성**

Mining 프로젝트의 불용어 파일 참고 + 보완

- [ ] **Step 2: text_mining.py 작성**

kiwipiepy 기반 한국어 텍스트 마이닝:

```python
from kiwipiepy import Kiwi

class TextMiner:
    def __init__(self):
        self.kiwi = Kiwi()
        self.stopwords = self._load_stopwords()

    def tokenize(self, texts, pos_filter=("NNG","NNP","VA","VV")):
        """형태소 분석 → 필터링된 토큰 리스트"""

    def tfidf_analysis(self, texts, top_n=20):
        """TF-IDF 키워드 추출"""

    def lda_topics(self, texts, n_topics=5, n_words=10):
        """LDA 토픽 모델링"""

    def keyword_network(self, texts, top_n=30, min_cooccurrence=2):
        """키워드 동시출현 네트워크 (networkx)"""

    def centrality_analysis(self, G):
        """중심성 분석 (degree, betweenness, eigenvector)"""

    def community_detection(self, G):
        """커뮤니티 탐지 (modularity 기반)"""

    def sentiment_network(self, texts):
        """긍·부정 키워드 네트워크 (감성사전 색상)"""

    def create_wordcloud(self, word_freq, color_func=None):
        """워드클라우드 생성"""

    def full_analysis(self, texts, chart_mode="plotly"):
        """전체 파이프라인: 형태소→TF-IDF→LDA→네트워크→감성→워드클라우드"""
```

- [ ] **Step 3: app.py에 /api/textmining 엔드포인트 추가**

옵션 파라미터:
- text_column: 텍스트 컬럼 (자동 탐지)
- pos_filter: 품사 필터 (명사/형용사/동사/부사 체크박스)
- n_topics: 토픽 수 (슬라이더 3~10)
- top_n_keywords: 키워드 수 (슬라이더 10~50)
- analyses: 실행할 분석 체크박스 (TF-IDF, LDA, 네트워크, 워드클라우드, 긍부정)

- [ ] **Step 4: tab_textmining.html**

- 옵션 패널 (품사 필터, 토픽 수, 분석 선택)
- TF-IDF 키워드 TOP N 테이블 + 막대 차트
- LDA 토픽별 키워드 테이블 + 토픽 점유율 차트
- 키워드 네트워크 그래프 (Plotly 인터랙티브 / matplotlib 정적)
- 중심성 분석 막대 차트
- 커뮤니티 탐지 결과
- 긍·부정 워드클라우드
- 전체 결과 CSV/ZIP 다운로드

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "feat: tab5 text mining"
```

### Task 8: 배포 설정

**Files:**
- Create: `Dockerfile`
- Create: `railway.toml`
- Create: `render.yaml`
- Create: `fonts/` (NanumGothic.ttf 포함)

- [ ] **Step 1: Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y fonts-nanum && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads results
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
```

핵심: Java 불필요, fonts-nanum만 설치

- [ ] **Step 2: railway.toml**

```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
healthcheckPath = "/"
restartPolicyType = "on_failure"
```

- [ ] **Step 3: render.yaml**

```yaml
services:
  - type: web
    name: review-analyzer
    runtime: docker
    plan: free
    healthCheckPath: /
    envVars:
      - key: PORT
        value: 8000
```

- [ ] **Step 4: 로컬 테스트**

```bash
cd ~/review-analyzer
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# 브라우저에서 http://localhost:8000 접속
# 각 탭에 테스트 파일 업로드하여 분석 확인
```

- [ ] **Step 5: 커밋 + 배포**

```bash
git add -A && git commit -m "feat: deployment config"
```

---

## 핵심 설계 원칙

1. **수업 순서 = 탭 순서** — 탭1(리뷰분류) → 탭2(감성) → 탭3(EDA) → 탭4(TCP/RFM) → 탭5(텍스트마이닝)
2. **차트 모드 자율 선택** — 모든 탭에 `인터랙티브(Plotly)` / `정적(이미지)` 토글
3. **자동 컬럼 탐지** — 수강생이 컬럼명 몰라도 동작
4. **Java 없는 한국어 NLP** — kiwipiepy만 사용
5. **Mining 프로젝트 UI 참고** — Bootstrap 5, 그라데이션 헤더, 카드 레이아웃
6. **각 탭 독립 모듈** — analyzer/ 아래 파일별 분리, 테스트 용이
7. **결과 다운로드** — CSV + PNG/HTML 차트 + ZIP 전체
