## ADDED Requirements

### Requirement: 多模型訓練與評估（LR/NB/SVM）
系統 SHALL 提供三種基線模型（Logistic Regression、Multinomial Naïve Bayes、Linear SVM），並可於 CLI 與 UI 中切換。

#### Scenario: CLI 選擇模型並訓練
- WHEN 執行 `spam-train --data <csv> --model-out <pkl> --model {lr|nb|svm}`
- THEN 成功以所選模型訓練並輸出模型檔與指標

#### Scenario: 評估與圖表輸出
- WHEN 執行 `spam-eval --data <csv> --model <pkl> --out-dir metrics/`
- THEN 產生 accuracy/precision/recall/f1 與混淆矩陣、ROC/PR 圖檔

### Requirement: Streamlit 互動介面
使用者 SHALL 能於 UI 上傳 CSV、選擇模型、啟動訓練並查看結果與圖表，亦可輸入句子進行預測。

#### Scenario: 上傳與訓練
- WHEN 於 UI 上傳含 `label,text` 的 CSV 並選擇模型
- THEN 顯示訓練完成之指標與圖表

#### Scenario: 單句預測
- WHEN 在 UI 輸入句子並提交
- THEN 顯示預測標籤與分數

### Requirement: 資料來源與文字前處理
系統 SHALL 能從本機資料夾或上傳取得資料，並完成基本清理與向量化。

#### Scenario: 從本機資料夾選擇或上傳
- WHEN 使用者在 UI 選擇 `dataset/`、`uploads/` 或 `data/` 內的 CSV，或上傳新檔（可保存至 `uploads/`）
- THEN 資料載入為兩欄 `label,text` 並進入訓練流程

#### Scenario: 文字清理與向量化
- WHEN 載入資料（無表頭 CSV 須依序為 label,text）
- THEN 進行移除網址、數字、標點與轉小寫
- AND 使用向量化轉換，支援：
  - TF‑IDF 與 Bag‑of‑Words（二擇一）：`--vec tfidf|bow`（UI 可切換）
  - n‑gram 範圍：`--ngram (1,1)|(1,2)|...`
  - Tokenize 規則（Regex）：`--token-pattern` 可設定
  - 停用詞：可選 `english` 或關閉

## MODIFIED Requirements

### Requirement: CLI 相容性與輸出位置
系統 SHALL 擴充 CLI 並維持相容預設。

#### Scenario: 模型參數相容預設
- WHEN 使用者未指定 `--model`
- THEN `spam-train|spam-eval|spam-predict` 預設使用 `svm`

#### Scenario: 指標與圖表輸出到 metrics/
- WHEN 未指定輸出路徑
- THEN 指標 CSV 與圖檔預設輸出到 `metrics/`

## REMOVED Requirements
（none）

