## ADDED Requirements

### Requirement: 基線垃圾郵件分類（SVM）
系統 SHALL 提供以線性 SVM 訓練之垃圾郵件分類模型，並能以 CLI 完成訓練、評估與預測。

#### Scenario: 成功訓練並輸出模型與指標
- WHEN 使用者執行 `spam-train --data <csv> --model-out <pkl>`
- AND `<csv>` 含欄位 `label,text`
- THEN 訓練完成並在 `<pkl>` 存檔
- AND 終端輸出 accuracy / precision / recall / f1 指標與混淆矩陣

#### Scenario: 支援多種標籤格式
- WHEN CSV 的 `label` 為 `{ham, spam}` 或 `{0,1}`
- THEN 系統自動正規化為二元標籤（0=ham, 1=spam）

#### Scenario: 預測單一句子
- WHEN 執行 `spam-predict --model <pkl> --text "win free prize"`
- THEN 輸出 `label` 與機率/分數

#### Scenario: 錯誤處理（資料欄位缺失）
- WHEN 訓練資料缺少 `label` 或 `text` 欄位
- THEN 以清楚錯誤訊息終止（描述缺失欄位）

#### Scenario: 可重現性
- WHEN 提供 `--seed <n>`
- THEN 相同資料與參數下多次訓練結果一致（指標波動在容忍範圍內）

## MODIFIED Requirements
（none）

## REMOVED Requirements
（none）

