"""이커머스 리뷰/데이터 분석 웹앱 — FastAPI 메인."""
import os
import uuid
import shutil
import json
import logging
import traceback

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd

# ---------------------------------------------------------------------------
# 앱 설정
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("review-analyzer")

app = FastAPI(title="이커머스 데이터 분석", description="수업용 리뷰/데이터 분석 웹앱")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------------
# 공용 유틸
# ---------------------------------------------------------------------------

def load_dataframe(file: UploadFile) -> tuple[pd.DataFrame, str]:
    """업로드 파일 → DataFrame + 저장 경로."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower()
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if ext == ".csv":
        df = pd.read_csv(save_path)
    else:
        df = pd.read_excel(save_path)
    return df, job_id


def detect_text_column(df: pd.DataFrame) -> str:
    """리뷰/텍스트 컬럼 자동 탐지."""
    priority = ["리뷰", "내용", "텍스트", "후기", "comment", "review", "content", "body", "text"]
    candidates = []
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        avg_len = sample.str.len().mean()
        candidates.append((col, avg_len))
    candidates.sort(key=lambda x: x[1], reverse=True)
    for col, _ in candidates:
        if any(kw in col.lower() for kw in priority):
            return col
    return candidates[0][0] if candidates else df.columns[0]


# ---------------------------------------------------------------------------
# 페이지 라우트
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# API 엔드포인트 — 탭1: 리뷰 분류
# ---------------------------------------------------------------------------

@app.post("/api/classify")
async def api_classify(file: UploadFile = File(...), chart_mode: str = Form("plotly")):
    try:
        from analyzer.classify import run_classification
        df, job_id = load_dataframe(file)
        result = run_classification(df, job_id, chart_mode)
        return JSONResponse({"status": "ok", "job_id": job_id, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# API 엔드포인트 — 탭2: 감성 분석
# ---------------------------------------------------------------------------

@app.post("/api/sentiment")
async def api_sentiment(file: UploadFile = File(...), chart_mode: str = Form("plotly")):
    try:
        from analyzer.sentiment import run_sentiment
        df, job_id = load_dataframe(file)
        result = run_sentiment(df, job_id, chart_mode)
        return JSONResponse({"status": "ok", "job_id": job_id, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# API 엔드포인트 — 탭3: EDA
# ---------------------------------------------------------------------------

@app.post("/api/eda")
async def api_eda(file: UploadFile = File(...), chart_mode: str = Form("plotly")):
    try:
        from analyzer.eda import run_eda
        df, job_id = load_dataframe(file)
        result = run_eda(df, job_id, chart_mode)
        return JSONResponse({"status": "ok", "job_id": job_id, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# API 엔드포인트 — 탭4: 커머스 분석
# ---------------------------------------------------------------------------

@app.post("/api/tcp")
async def api_tcp(
    file: UploadFile = File(...),
    chart_mode: str = Form("plotly"),
    dimensions: str = Form("time,customer,product"),
    date_col: str = Form(""),
    customer_col: str = Form(""),
    product_col: str = Form(""),
    quantity_col: str = Form(""),
    price_col: str = Form(""),
):
    try:
        from analyzer.tcp_rfm import run_tcp
        df, job_id = load_dataframe(file)
        col_map = {
            "date": date_col, "customer": customer_col,
            "product": product_col, "quantity": quantity_col, "price": price_col,
        }
        result = run_tcp(df, job_id, chart_mode, col_map, dimensions=dimensions.split(","))
        return JSONResponse({"status": "ok", "job_id": job_id, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# API 엔드포인트 — 탭5: 텍스트 마이닝
# ---------------------------------------------------------------------------

@app.post("/api/textmining")
async def api_textmining(
    file: UploadFile = File(...),
    chart_mode: str = Form("plotly"),
    n_topics: int = Form(5),
    top_n: int = Form(20),
    pos_filter: str = Form("NNG,NNP"),
    analyses: str = Form("tfidf,lda,network,wordcloud,sentiment"),
):
    try:
        from analyzer.text_mining import run_textmining
        df, job_id = load_dataframe(file)
        result = run_textmining(
            df, job_id, chart_mode,
            n_topics=n_topics, top_n=top_n,
            pos_filter=pos_filter.split(","),
            analyses=analyses.split(","),
        )
        return JSONResponse({"status": "ok", "job_id": job_id, **result})
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# 다운로드 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    path = os.path.join(RESULT_DIR, job_id, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    return FileResponse(path, filename=filename)


# ---------------------------------------------------------------------------
# 헬스체크
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
