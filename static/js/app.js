/* 이커머스 데이터 분석 — 프론트엔드 로직 */

document.addEventListener("DOMContentLoaded", () => {
    // 슬라이더 값 표시
    const topicSlider = document.getElementById("tm-topics");
    const topnSlider = document.getElementById("tm-topn");
    if (topicSlider) topicSlider.oninput = () => document.getElementById("tm-topics-val").textContent = topicSlider.value;
    if (topnSlider) topnSlider.oninput = () => document.getElementById("tm-topn-val").textContent = topnSlider.value;

    // 파일 업로드 영역
    document.querySelectorAll(".upload-zone").forEach(zone => {
        const input = zone.querySelector(".file-input");
        const nameDiv = zone.querySelector(".file-name");
        const tab = zone.dataset.tab;
        const btn = document.querySelector(`.btn-analyze[data-tab="${tab}"]`);

        zone.addEventListener("click", () => input.click());
        zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
        zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
        zone.addEventListener("drop", e => {
            e.preventDefault();
            zone.classList.remove("dragover");
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                onFileSelected(input, nameDiv, btn, tab);
            }
        });
        input.addEventListener("change", () => onFileSelected(input, nameDiv, btn, tab));
    });

    // 분석 버튼
    document.querySelectorAll(".btn-analyze").forEach(btn => {
        btn.addEventListener("click", () => runAnalysis(btn));
    });
});

function onFileSelected(input, nameDiv, btn, tab) {
    if (input.files.length) {
        nameDiv.textContent = input.files[0].name;
        btn.disabled = false;
        // TCP 탭: 컬럼 매핑 미리 로드
        if (tab === "tcp") preloadColumns(input.files[0]);
    }
}

async function preloadColumns(file) {
    const fd = new FormData();
    fd.append("file", file);
    // 간단히 첫 행만 읽어서 컬럼 목록 가져오기 (서버에 별도 엔드포인트 없으면 클라이언트에서 처리)
    // 여기서는 분석 시 서버가 자동 탐지하므로, 셀렉트 박스는 분석 결과에서 채움
    document.getElementById("tcp-columns").style.display = "block";
}

async function runAnalysis(btn) {
    const tab = btn.dataset.tab;
    const api = btn.dataset.api;
    const zone = document.querySelector(`.upload-zone[data-tab="${tab}"]`);
    const fileInput = zone.querySelector(".file-input");
    const chartMode = document.querySelector(`input[name="${tab}_chart"]:checked`)?.value || "plotly";

    if (!fileInput.files.length) return;

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("chart_mode", chartMode);

    // 탭별 추가 파라미터
    if (tab === "tcp") {
        ["date", "customer", "product", "quantity", "price"].forEach(k => {
            const sel = document.getElementById(`tcp-${k}-col`);
            if (sel) fd.append(`${k}_col`, sel.value || "");
        });
    }
    if (tab === "textmining") {
        fd.append("n_topics", document.getElementById("tm-topics")?.value || "5");
        fd.append("top_n", document.getElementById("tm-topn")?.value || "20");
        const posFilter = [...document.querySelectorAll(".pos-filter:checked")].map(c => c.value).join(",");
        fd.append("pos_filter", posFilter || "NNG,NNP");
        const analyses = [...document.querySelectorAll(".analysis-check:checked")].map(c => c.value).join(",");
        fd.append("analyses", analyses || "tfidf");
    }

    // 로딩
    showLoading(true, `${getTabName(tab)} 분석 중...`);
    btn.disabled = true;

    try {
        const resp = await fetch(api, { method: "POST", body: fd });
        const data = await resp.json();
        if (data.status === "ok") {
            renderResult(tab, data, chartMode);
        } else {
            alert("분석 오류: " + (data.message || "알 수 없는 오류"));
        }
    } catch (e) {
        alert("네트워크 오류: " + e.message);
    } finally {
        showLoading(false);
        btn.disabled = false;
    }
}

function getTabName(tab) {
    const names = { classify: "리뷰 분류", sentiment: "감성 분석", eda: "EDA", tcp: "TCP/RFM", textmining: "텍스트 마이닝" };
    return names[tab] || tab;
}

function showLoading(show, text) {
    const overlay = document.getElementById("loadingOverlay");
    const textEl = document.getElementById("loadingText");
    if (text) textEl.textContent = text;
    overlay.classList.toggle("show", show);
}

function renderResult(tab, data, chartMode) {
    const area = document.getElementById(`result-${tab}`);
    area.innerHTML = "";
    area.classList.add("show");

    // 요약 테이블
    if (data.summary_html) {
        area.innerHTML += `<div class="card"><div class="card-header">분석 결과 요약</div><div class="card-body">${data.summary_html}</div></div>`;
    }

    // 차트들
    if (data.charts) {
        data.charts.forEach((chart, i) => {
            const div = document.createElement("div");
            div.className = "chart-container";
            if (chart.title) {
                div.innerHTML += `<h6 class="mb-2">${chart.title}</h6>`;
            }
            if (chartMode === "plotly" && chart.plotly) {
                const plotDiv = document.createElement("div");
                plotDiv.id = `chart-${tab}-${i}`;
                div.appendChild(plotDiv);
                area.appendChild(div);
                Plotly.newPlot(plotDiv.id, chart.plotly.data, chart.plotly.layout, { responsive: true });
            } else if (chart.image) {
                div.innerHTML += `<img src="data:image/png;base64,${chart.image}" alt="${chart.title || '차트'}">`;
                area.appendChild(div);
            }
        });
    }

    // 대표 리뷰 / 상세 결과
    if (data.details_html) {
        area.innerHTML += `<div class="card"><div class="card-header">상세 결과</div><div class="card-body">${data.details_html}</div></div>`;
    }

    // 다운로드 버튼
    if (data.job_id && data.downloads) {
        let dlHtml = '<div class="card"><div class="card-header">다운로드</div><div class="card-body">';
        data.downloads.forEach(f => {
            dlHtml += `<a href="/api/download/${data.job_id}/${f.filename}" class="btn btn-download me-2 mb-2"><i class="fas fa-download"></i> ${f.label}</a>`;
        });
        dlHtml += "</div></div>";
        area.innerHTML += dlHtml;
    }

    // 스크롤
    area.scrollIntoView({ behavior: "smooth", block: "start" });
}
