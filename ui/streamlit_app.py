from __future__ import annotations

import io
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 讓 `ml` 套件可被匯入（以專案根目錄為基準）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.data import clean_text, normalize_label  # noqa: E402
from ml.features import make_vectorizer  # noqa: E402
from ml.models import get_model  # noqa: E402

try:
    from ml.metrics import compute_basic_metrics, threshold_sweep  # noqa: E402
except Exception:  # 相容處理：若熱重載尚未載到新版函式
    from ml.metrics import compute_basic_metrics  # type: ignore
    import numpy as _np
    from sklearn.metrics import confusion_matrix as _cm
    import pandas as _pd

    def threshold_sweep(y_true, y_score, thresholds=None):  # type: ignore
        y_true = _np.asarray(y_true)
        y_score = _np.asarray(y_score)
        if thresholds is None:
            lo, hi = float(_np.min(y_score)), float(_np.max(y_score))
            if not _np.isfinite(lo) or not _np.isfinite(hi) or lo == hi:
                thresholds = _np.linspace(0.0, 1.0, 51)
            else:
                thresholds = _np.linspace(lo, hi, 51)
        rows = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            cm = _cm(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            tpr = rec
            fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)
            rows.append(
                {
                    "threshold": float(t),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                }
            )
        return _pd.DataFrame(rows)


st.set_page_config(page_title="Spam Classification Demo", page_icon="📨", layout="centered")


def parse_csv(file_bytes: bytes, no_header: bool) -> pd.DataFrame:
    if no_header:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, names=["label", "text"], encoding_errors="ignore")
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding_errors="ignore")
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        label_col = "label" if "label" in cols else ("category" if "category" in cols else "target")
        text_col = "text" if "text" in cols else ("message" if "message" in cols else "sms")
        df = df[[label_col, text_col]].copy()
        df.columns = ["label", "text"]
    df["label"] = df["label"].map(normalize_label)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df


def build_vectorizer(vec_kind: str, ngram: Tuple[int, int], token_pattern: str, stop_words_opt: Optional[str]):
    # 優先使用專案封裝；如簽名不同則回退到 sklearn 直接建立
    try:
        return make_vectorizer(kind=vec_kind, ngram=ngram, token_pattern=token_pattern, stop_words=stop_words_opt)
    except TypeError:  # 舊版簽名相容
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        common = dict(ngram_range=ngram, token_pattern=token_pattern, stop_words=stop_words_opt, lowercase=False)
        return CountVectorizer(**common) if vec_kind == "bow" else TfidfVectorizer(**common)


def list_existing_csvs() -> List[str]:
    candidates: List[str] = []
    for root in ["dataset", "uploads", "data"]:
        if os.path.isdir(root):
            for name in os.listdir(root):
                if name.lower().endswith(".csv"):
                    candidates.append(os.path.join(root, name))
    return candidates


def main():
    st.title("📨 Spam Classification Demo")
    st.write("上傳 CSV（label,text）或從 dataset/uploads/data 選擇現有檔案，選擇模型並訓練。")

    with st.sidebar:
        st.header("設定")
        model_kind = st.selectbox("模型", ["svm", "lr", "nb"], index=0)
        vec_kind = st.selectbox("向量器", ["tfidf", "bow"], index=0)
        ngram_max = st.slider("n-gram 上限", min_value=1, max_value=3, value=1)
        token_pattern = st.text_input("Token Pattern (regex)", value=r"(?u)\b\w+\b")
        use_stop = st.checkbox("使用英文停用詞", value=False)
        test_split = st.slider("測試集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)
        no_header = st.checkbox("無表頭資料（Packt CSV）", value=True)

    source = st.radio("資料來源", ["上傳檔案", "從現有檔案選擇"], horizontal=True)

    df: Optional[pd.DataFrame] = None
    if source == "上傳檔案":
        uploaded = st.file_uploader("上傳 CSV 檔 (label,text)", type=["csv"])
        save_upload = st.checkbox("將上傳檔保存至 uploads/ 供他人使用", value=True)
        if uploaded is not None:
            file_bytes = uploaded.read()
            df = parse_csv(file_bytes, no_header=no_header)
            st.success(f"資料已載入：{len(df)} 筆（上傳）")
            st.dataframe(df.head())
            if save_upload:
                os.makedirs("uploads", exist_ok=True)
                out_path = os.path.join("uploads", uploaded.name)
                with open(out_path, "wb") as w:
                    w.write(file_bytes)
                st.write(f"已保存到 {out_path}")
    else:
        options = list_existing_csvs()
        if options:
            sel = st.selectbox("選擇現有 CSV 檔案", options)
            if sel:
                with open(sel, "rb") as f:
                    df = parse_csv(f.read(), no_header=no_header)
                st.success(f"資料已載入：{len(df)} 筆（{sel}）")
                st.dataframe(df.head())
        else:
            st.info("尚未發現可用 CSV，請先放入 dataset/ 或 data/，或改用上傳模式。")

    # 自動訓練：一旦有 df 就直接訓練與顯示結果
    if df is not None:
        with st.spinner("訓練中..."):
            from sklearn.metrics import confusion_matrix as _confusion_matrix
            from sklearn.metrics import roc_curve as _roc_curve, auc as _auc, precision_recall_curve as _prc
            vec = build_vectorizer(vec_kind, (1, ngram_max), token_pattern, ("english" if use_stop else None))
            pipe = Pipeline([("vec", vec), ("model", get_model(model_kind))])
            X = df["text"]; y = df["label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed, stratify=y)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if hasattr(pipe.named_steps["model"], "decision_function"):
                y_score = pipe.decision_function(X_test)
            elif hasattr(pipe.named_steps["model"], "predict_proba"):
                y_score = pipe.predict_proba(X_test)[:, 1]
            else:
                y_score = y_pred
            metrics = compute_basic_metrics(y_test, y_pred)

        st.subheader("指標")
        st.json(metrics)

        # 混淆矩陣
        import matplotlib.pyplot as plt

        st.subheader("混淆矩陣")
        cm = _confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.figure.colorbar(im, ax=ax_cm)
        ax_cm.set(xlabel="Predicted", ylabel="True")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

        # ROC / PR
        try:
            y_score_arr = np.asarray(y_score)
            if y_score_arr.ndim == 1:
                st.subheader("ROC / PR")
                fpr, tpr, _ = _roc_curve(y_test, y_score_arr)
                roc_auc = _auc(fpr, tpr)
                fig1, ax1 = plt.subplots()
                ax1.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
                ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
                ax1.legend()
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

                p, r, _ = _prc(y_test, y_score_arr)
                ap = _auc(r, p)
                fig2, ax2 = plt.subplots()
                ax2.plot(r, p, label=f"AP={ap:.3f}")
                ax2.legend()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

                # Threshold sweep + interactive threshold
                st.subheader("Threshold Sweep")
                df_thr = threshold_sweep(y_test, y_score_arr)
                st.line_chart(df_thr.set_index("threshold")[["precision", "recall", "f1"]])

                thr_min = float(df_thr["threshold"].min())
                thr_max = float(df_thr["threshold"].max())
                thr_default = float(np.clip(np.median(y_score_arr), thr_min, thr_max))
                step = max((thr_max - thr_min) / 100.0, 0.001)
                thr = st.slider("Decision threshold", min_value=thr_min, max_value=thr_max, value=thr_default, step=step)

                y_pred_thr = (y_score_arr >= thr).astype(int)
                st.caption("基於目前門檻的指標與混淆矩陣：")
                metrics_thr = compute_basic_metrics(y_test, y_pred_thr)
                st.json(metrics_thr)

                cm2 = _confusion_matrix(y_test, y_pred_thr)
                fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 4))
                im2 = ax_cm2.imshow(cm2, cmap="Purples")
                ax_cm2.figure.colorbar(im2, ax=ax_cm2)
                ax_cm2.set(xlabel="Predicted", ylabel="True", title=f"Threshold = {thr:.3f}")
                t2 = cm2.max() / 2.0
                for i in range(cm2.shape[0]):
                    for j in range(cm2.shape[1]):
                        ax_cm2.text(j, i, int(cm2[i, j]), ha="center", va="center", color="white" if cm2[i, j] > t2 else "black")
                st.pyplot(fig_cm2, use_container_width=True)
                plt.close(fig_cm2)
        except Exception:
            pass

        # Top Tokens by Class
        try:
            st.subheader("Top Tokens by Class")
            top_n = st.slider("Top-N tokens", min_value=5, max_value=100, value=30, step=5)
            from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer

            cnt_vec = _CountVectorizer(
                ngram_range=(1, ngram_max),
                token_pattern=token_pattern,
                stop_words=("english" if use_stop else None),
                lowercase=False,
            )
            X_all = cnt_vec.fit_transform(df["text"])  # 使用目前載入資料
            vocab = np.array(cnt_vec.get_feature_names_out())
            labels_arr = df["label"].to_numpy()
            ham_mask = labels_arr == 0
            spam_mask = labels_arr == 1
            ham_counts = np.asarray(X_all[ham_mask].sum(axis=0)).ravel() if ham_mask.any() else np.zeros(X_all.shape[1])
            spam_counts = np.asarray(X_all[spam_mask].sum(axis=0)).ravel() if spam_mask.any() else np.zeros(X_all.shape[1])

            def _top_tokens(counts: np.ndarray, k: int):
                if counts.sum() == 0:
                    return pd.DataFrame({"token": [], "frequency": []})
                idx = np.argsort(counts)[::-1][:k]
                return pd.DataFrame({"token": vocab[idx], "frequency": counts[idx]})

            ham_top = _top_tokens(ham_counts, top_n)
            spam_top = _top_tokens(spam_counts, top_n)
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Class: ham")
                if len(ham_top) == 0:
                    st.info("無資料可顯示（ham 類別無樣本或全為停用詞）")
                else:
                    st.bar_chart(ham_top.set_index("token")[ ["frequency"] ])
            with col2:
                st.caption("Class: spam")
                if len(spam_top) == 0:
                    st.info("無資料可顯示（spam 類別無樣本或全為停用詞）")
                else:
                    st.bar_chart(spam_top.set_index("token")[ ["frequency"] ])
            st.caption("頻率軸為該類別內 token 出現次數；token 已經過前處理（小寫與清理）。")
        except Exception:
            pass

        st.subheader("即時預測")
        text = st.text_input("輸入句子進行預測")
        if text:
            pred = int(pipe.predict([text])[0])
            st.write(f"預測標籤：{'spam' if pred == 1 else 'ham'}")


if __name__ == "__main__":
    main()

