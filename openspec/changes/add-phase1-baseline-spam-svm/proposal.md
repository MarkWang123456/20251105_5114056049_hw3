# Change: Phase1 基線 — SVM 垃圾郵件分類器

## Why
建立可運行的垃圾郵件分類「基線模型」，支援後續 Phase2/Phase3 的強化（如調參、服務化、監控），先聚焦於資料讀取、特徵抽取、訓練與評估。

## What Changes
- 新增資料讀取與清洗：支援 CSV（欄位 `label,text`），`label` 可為 `{spam, ham}` 或 `{1,0}`，內建標籤正規化。
- 新增特徵工程：TF‑IDF（支援 unigram；bigram 作為可選）、停用詞、最小字頻與最大字典大小。
- 新增模型：線性 SVM（`LinearSVC`），`class_weight='balanced'`，可設定 `random_state`。
- 新增 CLI：
  - 訓練：`spam-train --data <path> --model-out <path> [--test-split 0.2] [--seed 42]`
  - 評估：`spam-eval --data <path> --model <path>`
  - 預測：`spam-predict --model <path> --text "free coupons..."` 或由 stdin/檔案批次
- 產出成果：模型檔（joblib/pkl）、指標（accuracy / precision / recall / f1）、混淆矩陣報表。
- 設定：資料來源預設 `data/spam.csv`，亦可由 `--data` 指定；不內嵌資料集。

## Impact
- Specs：新增能力 `spam-filtering`
- Code：新增 `spam/` 套件、CLI 入口、需求檔；不變更既有功能

## Out of Scope for Phase1
- 參數搜尋、交叉驗證、服務化 API、持續訓練/監控、資料漂移偵測（留待 Phase2+）

