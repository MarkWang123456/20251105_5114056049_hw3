# Change: 核心機器學習管線與 Streamlit UI

## Why
在 Phase1 基線（SVM）之上，補齊常見基線模型（Logistic Regression、Naïve Bayes、Linear SVM）與資料前處理管線，並提供可互動的 Streamlit UI，便於教學展示與快速試驗。

## What Changes
- 數據前處理與資料取得：
  - 自動下載/載入 Packt 提供之 `sms_spam_no_header.csv`（如本地不存在則下載）
  - 清理：移除網址、數字、標點，統一小寫
  - TF‑IDF：支援 n-gram、停用詞與字典大小參數化
- 多模型支援：`--model {lr,nb,svm}` 選擇 Logistic Regression / Naïve Bayes / Linear SVM
- 評估與輸出：accuracy、precision、recall、f1、混淆矩陣、ROC/PR 曲線；將指標 CSV 與圖檔輸出到 `metrics/`
- Streamlit UI：
  - 上傳 CSV（`label,text`）
  - 選擇模型與參數，啟動訓練
  - 顯示指標與圖表；提供單句或批次文字預測
- CLI 強化：訓練/評估/預測皆支援選擇模型、輸出指標與圖表到指定資料夾
 - 部署：提供 Streamlit Cloud 的部署指引，並於 README 記錄公開 Demo 網址

## Impact
- Specs：擴充 `spam-filtering` 能力；新增 `ui` 能力（互動展示）
- Code：新增 `ui/streamlit_app.py`、`ml/`（data/features/models）與 `cli/` 入口

## Out of Scope (本次不含)
- 超參數搜尋與交叉驗證、持續訓練與監控、雲端部署
