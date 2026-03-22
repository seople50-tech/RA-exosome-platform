# app_safe.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import io

# =====================
# UI 優化：背景 + 標題
# =====================
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)

st.title("🧬 RA Exosome AI Platform")
st.caption("Clinical + miRNA Analysis System")

# =====================
# CSV 上傳
# =====================
st.header("📂 上傳臨床資料 CSV")
uploaded_file = st.file_uploader("請上傳 CSV 檔案", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("原始資料")
    st.dataframe(df.head())

    # =====================
    # AI 分析示範：風險計算
    # =====================
    st.subheader("🧠 風險評分計算")
    if 'miRNA_score' not in df.columns:
        df['miRNA_score'] = np.random.rand(len(df)) * 10  # 生成示範數據

    df['risk_score'] = df['miRNA_score']
    df['risk_level'] = df['risk_score'].apply(lambda x: "High" if x > 5 else "Low")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("平均風險分數", round(df["risk_score"].mean(), 2))
    with col2:
        st.metric("高風險人數", (df["risk_level"]=="High").sum())

    st.subheader("風險資料表")
    st.dataframe(df[['patient_id','risk_score','risk_level']])

    # =====================
    # Heatmap (安全顯示)
    # =====================
    st.subheader("📊 Heatmap")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("數值欄位不足，無法生成 Heatmap")

    # =====================
    # PCA (安全顯示)
    # =====================
    st.subheader("🔹 PCA 分析")
    if len(numeric_cols) >= 2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(df[numeric_cols])
        df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
        st.write(df_pca.head())

        fig, ax = plt.subplots(figsize=(6,4))
        scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df['risk_score'], cmap='coolwarm')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Risk Score')
        st.pyplot(fig)
    else:
        st.info("數值欄位不足，無法做 PCA")

    # =====================
    # PDF 報告下載
    # =====================
    st.subheader("📄 下載 PDF 報告")
    def generate_pdf(df):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_file.name)
        styles = getSampleStyleSheet()
        content = [Paragraph("Clinical AI Analysis Report", styles["Title"])]
        for i, row in df.iterrows():
            text = f"{row.get('patient_id','NA')} - Risk: {row['risk_level']} (Score: {row['risk_score']:.2f})"
            content.append(Paragraph(text, styles["Normal"]))
        doc.build(content)
        return temp_file.name

    pdf_file = generate_pdf(df)
    with open(pdf_file, "rb") as f:
        st.download_button("📄 下載 PDF 報告", f, "report.pdf")
