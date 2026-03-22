import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# =======================
# UI 美化
# =======================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 RA Exosome AI Platform")
st.caption("Clinical + miRNA Analysis System")

# =======================
# CSV 上傳
# =======================
uploaded_file = st.file_uploader("📂 上傳 CSV 檔案", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("原始資料預覽")
    st.dataframe(df.head())

    # =======================
    # 風險分數計算
    # =======================
    st.subheader("🧠 風險評分計算")
    required_cols = ['glucose','cholesterol','crp','esr']  # 假設這些欄位存在
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"缺少欄位: {col}，風險計算將忽略此欄位")
    
    # 簡單範例計算分數
    def risk_score(row):
        score = 0
        if 'glucose' in row and (row['glucose']<70 or row['glucose']>100):
            score += 2
        if 'cholesterol' in row and (row['cholesterol']<125 or row['cholesterol']>200):
            score += 1
        if 'crp' in row and row['crp']>3:
            score += 3
        if 'esr' in row and row['esr']>20:
            score += 2
        return score

    df['risk_score'] = df.apply(risk_score, axis=1)
    df['risk_level'] = df['risk_score'].apply(lambda x: "High" if x>5 else "Low")

    col1, col2 = st.columns(2)
    col1.metric("平均風險分數", round(df["risk_score"].mean(),2))
    col2.metric("高風險人數", (df["risk_level"]=="High").sum())

    st.subheader("風險資料表")
    st.dataframe(df[['patient_id','risk_score','risk_level']])

    # =======================
    # Heatmap
    # =======================
    st.subheader("📊 Heatmap")
    plt.figure(figsize=(6,4))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)
    plt.savefig("heatmap.png")

    # =======================
    # PCA
    # =======================
    st.subheader("🔹 PCA 分析")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols)>=2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(df[numeric_cols])
        df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
        st.write(df_pca.head())

        plt.figure(figsize=(6,4))
        scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df['risk_score'], cmap='coolwarm')
        plt.colorbar(scatter, label='Risk Score')
        st.pyplot(plt)
        plt.savefig("pca.png")
    else:
        st.info("數值欄位不足，無法做 PCA")

    # =======================
    # Top miRNA 條形圖（假設有 miRNA 欄位）
    # =======================
    mirna_cols = [col for col in df.columns if 'miRNA' in col]
    if len(mirna_cols)>0:
        st.subheader("📈 Top miRNA 條形圖")
        top_mirna = df[mirna_cols].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        top_mirna.plot(kind='bar', ax=ax)
        ax.set_ylabel("平均表達量")
        st.pyplot(fig)
        fig.savefig("top_mirna.png")

    # =======================
    # PDF 報告
    # =======================
    st.subheader("📄 下載 PDF 報告")
    def generate_pdf(df):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_file.name)
        styles = getSampleStyleSheet()
        content = [Paragraph("RA Exosome Analysis Report", styles["Title"])]
        for i,row in df.iterrows():
            text = f"{row.get('patient_id','NA')} - Risk: {row['risk_level']} (Score: {row['risk_score']:.2f})"
            content.append(Paragraph(text, styles["Normal"]))
        # 加圖片
        try:
            content.append(Spacer(1,12))
            content.append(Image("heatmap.png", width=400, height=200))
            content.append(Spacer(1,12))
            content.append(Image("pca.png", width=400, height=200))
            if len(mirna_cols)>0:
                content.append(Spacer(1,12))
                content.append(Image("top_mirna.png", width=400, height=200))
        except:
            pass
        doc.build(content)
        return temp_file.name

    pdf_file = generate_pdf(df)
    with open(pdf_file,"rb") as f:
        st.download_button("📄 下載 PDF 報告", f, "RA_analysis_report.pdf")
