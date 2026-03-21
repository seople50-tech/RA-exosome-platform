import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

st.set_page_config(page_title="AI Clinical + Exosome Analysis", layout="wide")

st.title("🧬 AI 臨床 + 外泌體分析平台")
st.write("上傳 CSV，自動分析臨床指標 + miRNA 特徵")

uploaded_file = st.file_uploader("📂 上傳 CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =====================
    # 1️⃣ 臨床風險分析
    # =====================
    st.subheader("🩺 臨床風險評估")

    def risk_score(row):
        score = 0
        if row["glucose"] < 70 or row["glucose"] > 100:
            score += 2
        if row["cholesterol"] < 125 or row["cholesterol"] > 200:
            score += 1
        if row["crp"] > 3:
            score += 3
        if row["esr"] > 20:
            score += 2
        return score

    df["risk_score"] = df.apply(risk_score, axis=1)

    def risk_level(score):
        if score <= 2:
            return "Low"
        elif score <= 5:
            return "Medium"
        else:
            return "High"

    df["risk_level"] = df["risk_score"].apply(risk_level)

    st.dataframe(df[["patient_id", "risk_score", "risk_level"]])

    # =====================
    # 2️⃣ PCA 分析
    # =====================
    st.subheader("🧬 miRNA PCA 分析")

    mirna_cols = [col for col in df.columns if "miRNA" in col]
    X = df[mirna_cols]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    df["pca1"] = pca_result[:,0]
    df["pca2"] = pca_result[:,1]

    fig, ax = plt.subplots()
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        ax.scatter(subset["pca1"], subset["pca2"], label=f"Label {label}")
    ax.legend()
    ax.set_title("PCA of miRNA")

    st.pyplot(fig)

    # =====================
    # 3️⃣ Top miRNA 分析
    # =====================
    st.subheader("🔥 關鍵 miRNA 特徵")

    model = RandomForestClassifier()
    model.fit(X, df["label"])

    importances = pd.Series(model.feature_importances_, index=mirna_cols)
    top10 = importances.sort_values(ascending=False).head(10)

    st.bar_chart(top10)

    # =====================
    # 4️⃣ Heatmap
    # =====================
    st.subheader("🌡️ miRNA Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.heatmap(df[mirna_cols].iloc[:20], cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # =====================
    # 5️⃣ 下載報表
    # =====================
    st.subheader("📥 下載報表")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("下載完整分析 CSV", csv, "analysis_report.csv")