## 1. 資料與特徵
- [ ] 1.1 定義 CSV 介面與標籤正規化（支援 `{ham, spam}`/`{0,1}`）
- [ ] 1.2 文本前處理：基本清洗（大小寫、空白、可選移除符號）
- [ ] 1.3 建立 TF‑IDF 管線（unigram，bigram 可選、停用詞、字典大小/最小字頻）

## 2. 模型與訓練
- [ ] 2.1 使用 `LinearSVC` + `class_weight='balanced'`
- [ ] 2.2 資料切分（train/test；`--test-split`）
- [ ] 2.3 訓練與評估（accuracy / precision / recall / f1、混淆矩陣）
- [ ] 2.4 模型保存（joblib/pickle）與載入

## 3. CLI 與使用者體驗
- [ ] 3.1 `spam-train`：參數 `--data --model-out [--test-split] [--seed]`
- [ ] 3.2 `spam-eval`：參數 `--data --model`
- [ ] 3.3 `spam-predict`：參數 `--model --text/--file`；輸出標籤與分數
- [ ] 3.4 錯誤訊息與說明（缺欄位、檔案不存在）

## 4. 品質與文件
- [ ] 4.1 最小單元測試：資料載入與標籤正規化、TF‑IDF 綁定
- [ ] 4.2 README 片段：如何準備 `data/spam.csv`、如何訓練/評估/預測
- [ ] 4.3 `requirements.txt` 或 `pyproject.toml`：`scikit-learn`, `pandas`, `numpy`, `joblib`

## 5. Phase2（保留）
- [ ] 5.1 （空）

