"""python -m reviewapp 또는 review-analyzer 명령으로 서버 시작."""
import os
import uvicorn


def main():
    port = int(os.getenv("PORT", 8000))
    print(f"리뷰 분석기 서버 시작: http://localhost:{port}")
    uvicorn.run("reviewapp.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
