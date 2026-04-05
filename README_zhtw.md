# Gemma-Meeting-Notes

> **以 Gemma 4 E4B / E2B 驅動的會議音檔轉錄服務** — 上傳錄音檔，自動產生完整逐字稿與 SRT 字幕。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonliu-ai-product/Gemma-Meeting-Notes/blob/main/colab_demo.ipynb)

---

## 專案簡介

| | |
|---|---|
| **模型** | Gemma 4 E4B（4.5B 有效參數），可切換 E2B |
| **功能** | 音訊 → 文字（ASR 自動語音辨識） |
| **切片上限** | 每次推論 25 秒（Gemma 4 音訊限制 30 秒） |
| **輸出格式** | 純文字 + 含時間戳的 SRT 字幕檔 |
| **介面語言** | 繁體中文 / 英文（可切換） |
| **Demo 環境** | Google Colab（T4 GPU） |

---

## 架構說明

```
┌─────────────────────────────┐
│   前端  (index.html)         │  雙語介面 — 音檔上傳、進度顯示、結果下載
└────────────┬────────────────┘
             │ POST /api/transcribe
┌────────────▼────────────────┐
│   後端  (FastAPI :8000)      │  音訊切片、任務協調、結果合併
└────────────┬────────────────┘
             │ POST /transcribe（每個切片）
┌────────────▼────────────────┐
│  模型服務 (Flask :8001)      │  透過 HuggingFace Transformers 載入 Gemma 4 E4B
└─────────────────────────────┘
```

前端由 FastAPI 後端直接 serve 靜態檔案，不需額外啟動前端服務。

---

## 檔案結構

```
Gemma-Meeting-Notes/
├── backend/
│   ├── model_service.py   # 載入 Gemma 4 E4B，提供 /transcribe 端點
│   ├── main.py            # FastAPI — 處理上傳、切片、整合
│   ├── transcriber.py     # 音訊切片（pydub）+ SRT 時間戳建構
│   └── requirements.txt
├── frontend/
│   └── index.html         # 單一檔案雙語前端
└── colab_demo.ipynb       # 一鍵 Colab Demo
```

---

## 快速開始（Google Colab）

1. 點擊上方 **Open in Colab** 徽章
2. 切換 Runtime 至 **GPU**（Runtime → Change runtime type → T4 GPU）
3. 依序執行各 Cell：

| Cell | 步驟 | 說明 |
|------|------|------|
| 0 | GPU 確認 | 未偵測到 GPU 時會提示切換 |
| 1 | 安裝 ffmpeg | 系統音訊處理工具 |
| 2 | Clone 專案 | 從 GitHub 下載所有檔案 |
| 3 | 安裝 Python 依賴 | transformers、fastapi 等 |
| 4 | HuggingFace 登入 | Gemma 為 gated model，需先至 HF 同意授權 |
| 5 | 啟動模型服務 | **首次需下載 ~9 GB 模型，請耐心等候** |
| 6 | 啟動後端 API | 確認 Cell 5 成功後再執行 |
| 7 | 開啟 Demo 視窗 | Colab 自動彈出前端介面分頁 |
| 8 | 停止所有服務 | Demo 結束後執行 |

---

## 本機開發

```bash
# 1. 安裝 ffmpeg
brew install ffmpeg   # macOS
# 或：apt-get install ffmpeg

# 2. 安裝 Python 依賴
pip install -r backend/requirements.txt

# 3. 啟動模型服務
GEMMA_MODEL_ID=google/gemma-4-e4b-it python backend/model_service.py

# 4. 啟動後端（另開終端機）
MODEL_SERVICE_URL=http://localhost:8001 \
  uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend

# 5. 開啟瀏覽器 http://localhost:8000
```

---

## 切換模型大小

修改 Notebook Cell 5 的 `GEMMA_MODEL_ID`（或本機的環境變數）：

| 版本 | Model ID | 所需 VRAM |
|------|----------|-----------|
| E4B（預設） | `google/gemma-4-e4b-it` | 約 9 GB |
| E2B | `google/gemma-4-e2b-it` | 約 5 GB |

---

## 支援音檔格式

`mp3` · `wav` · `m4a` · `ogg` · `flac` · `webm`

長音檔會自動切割為 ≤ 25 秒的片段分批轉錄，最後合併輸出。

---

## 授權

Apache 2.0，詳見 [LICENSE](LICENSE)。  
Gemma 模型權重須遵守 [Google Gemma 使用條款](https://ai.google.dev/gemma/terms)。
