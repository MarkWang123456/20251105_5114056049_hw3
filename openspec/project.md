# Project Context

## Purpose
建立一個可複現的垃圾郵件（Spam）分類專案，提供完整的離線訓練與評估流程，以及可互動的 Streamlit 介面用於上傳資料、訓練模型、觀察指標與視覺化，作為後續階段（Phase2+）強化的基線。

## Tech Stack
- 語言：Python 3.10+
- 機器學習：scikit-learn（Logistic Regression、Naïve Bayes、Linear SVM）
- 資料處理：pandas、numpy
- 特徵工程：TF‑IDF、可選 n-gram、停用詞
- 介面：Streamlit（互動式訓練/預測）
- 視覺化：matplotlib / seaborn（混淆矩陣、ROC、PR）
- 模型保存：joblib/pickle
- CLI：argparse 或 Typer（訓練、評估、預測）

## Project Conventions

### Code Style
- Black + isort 格式化；flake8/ruff 做靜態檢查
- 命名遵循 pep8，模組/檔名使用蛇形命名，常數全大寫
- 目錄結構分離資料、程式、模型、UI 與 openspec

### Architecture Patterns
- 分層模組化：`data`（載入/清洗）→ `features`（向量化）→ `models`（訓練/推論）→ `cli`/`ui`
- 可組合的管線：`(clean → tokenize → tfidf) + (model)`
- 以 CLI 參數控制：資料路徑、測試切分、隨機種子、模型類型、超參數

### Testing Strategy
- 單元測試（pytest）：標籤正規化、向量化、單一/批次推論
- 簡易整合測試：小型樣本資料 end-to-end（train→eval→predict）
- 指標驗證：accuracy/precision/recall/f1 至少能在示例資料達到合理範圍

### Git Workflow
- trunk + feature branches；PR 必要
- Conventional Commits（feat/fix/docs/refactor/test/chore）
- 變更先以 OpenSpec 提案審核，再開始實作

## Domain Context
- 垃圾郵件分類，輸入為文字（email subject/body），輸出為二元標籤（ham/spam）
- 常見資料集：`label,text` 欄位；標籤可能為 {ham, spam} 或 {0,1}
- 英文文本為主，簡易英文停用詞表即可；中文語料不在本階段範圍

## Important Constraints
- 不內嵌大型資料集；預設讀取 `data/spam.csv`（本地自備）
- 可離線運行；訓練可在一般筆電完成
- 模型與指標輸出需可重現（支援 `--seed`）

## External Dependencies
- scikit-learn、pandas、numpy、joblib、streamlit、matplotlib、seaborn
- 若需下載停用詞或語言資源，採選用且離線可用的清單
# Project Context

## Purpose
建立一個可複現的 Email / SMS 垃圾郵件（Spam）分類專案，提供完整的離線訓練與評估流程，以及可互動的 Streamlit 介面用於上傳資料、訓練模型、觀察指標與視覺化，作為後續階段（Phase2+）強化的基線。

## Tech Stack
- 語言：Python 3.10+
- 機器學習：scikit-learn（Logistic Regression、Multinomial Naïve Bayes、Linear SVM）
- 資料處理：pandas、numpy
- 特徵工程：TF‑IDF / Bag‑of‑Words，支援 n‑gram、停用詞、Token Pattern
- 介面：Streamlit（互動式訓練/預測）
- 視覺化：matplotlib / seaborn（混淆矩陣、ROC、PR、Threshold Sweep）
- 模型保存：joblib/pickle
- CLI：argparse（訓練、評估、預測）

## Project Conventions

### Code Style
- Black + isort 格式化；flake8/ruff 做靜態檢查
- 命名遵循 pep8，模組/檔名使用蛇形命名，常數全大寫
- 目錄結構分離資料、程式、模型、UI 與 openspec

### Architecture Patterns
- 分層模組化：`data`（載入/清洗）→ `features`（向量化）→ `models`（訓練/推論）→ `cli`/`ui`
- 可組合的管線：`(clean → tokenize → vectorize) + (model)`
- 以 CLI 參數控制：資料路徑、測試切分、隨機種子、模型類型、向量化與超參數

### Testing Strategy
- 單元測試（pytest）：標籤正規化、向量化、單一/批次推論
- 簡易整合測試：小型樣本資料 end‑to‑end（train→eval→predict）
- 指標驗證：accuracy/precision/recall/f1 至少能在示例資料達到合理範圍

### Git Workflow
- trunk + feature branches；PR 必要
- Conventional Commits（feat/fix/docs/refactor/test/chore）
- 變更先以 OpenSpec 提案審核，再開始實作

## Domain Context
- 垃圾郵件分類，輸入為文字（email/SMS），輸出為二元標籤（ham/spam）
- 常見資料集：`label,text` 欄位；標籤可能為 {ham, spam} 或 {0,1}
- 英文文本為主；中文語料不在本階段範圍

## Important Constraints
- 不內嵌大型資料集；預設從 `dataset/` 或 `data/` 載入本機 CSV（或 UI 上傳）
- 可離線運行；訓練可在一般筆電完成
- 模型與指標輸出需可重現（支援 `--seed`）

## External Dependencies
- scikit-learn、pandas、numpy、joblib、streamlit、matplotlib、seaborn
- 若需停用詞資源，採選用且離線可用的清單
