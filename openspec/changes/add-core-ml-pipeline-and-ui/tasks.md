## 1. 數據與特徵
- [ ] 1.1 讀取本機 CSV（dataset/、uploads/、data/）或 UI 上傳（可保存到 uploads/）
- [ ] 1.2 標籤正規化（{ham,spam}/{0,1} → 0/1）
- [ ] 1.3 清理：移除網址、數字、標點，統一小寫；英文 tokenization
- [ ] 1.4 向量化選項：TF‑IDF / Bag‑of‑Words、n‑gram、Token Pattern、停用詞

## 2. 模型與訓練
- [ ] 2.1 Logistic Regression / Multinomial Naïve Bayes / Linear SVM（`--model` 切換）
- [ ] 2.2 資料切分與重現性（`--test-split`、`--seed`）
- [ ] 2.3 指標與可視化（accuracy/precision/recall/f1、混淆矩陣、ROC/PR）
- [ ] 2.4 模型保存/載入（joblib）
- [ ] 2.5 匯出：`metrics/` 輸出 metrics.csv 與圖檔（confusion_matrix.png、roc.png、pr.png）

## 3. CLI 與 UI
- [ ] 3.1 `spam-train`/`spam-eval`/`spam-predict` 支援 `--model {lr,nb,svm}`
- [ ] 3.2 `--vec {tfidf,bow}`、`--ngram`、`--token-pattern`、`--stop-words` 參數
- [ ] 3.3 UI 側欄：模型/向量器/n‑gram/Token Pattern/停用詞切換
- [ ] 3.4 指標與圖表預設輸出 `metrics/`（可用 `--out-dir` 覆蓋）
- [ ] 3.5 CLI 相容性：未指定 `--model` 時預設 `svm`（維持 Phase1 行為）

## 4. 文件與品質
- [ ] 4.1 README：安裝、資料格式、CLI 範例、UI 啟動方式
- [ ] 4.2 `requirements.txt`：sklearn, pandas, numpy, joblib, streamlit, matplotlib, seaborn
- [ ] 4.3 基礎測試：小樣本 e2e（train→eval→predict）
- [ ] 4.4 部署指引：Streamlit Cloud 部署步驟與公開 Demo 網址填寫位置

