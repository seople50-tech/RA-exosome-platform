import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import plotly.express as px

# =====================
# UI 美化
# =====================
st.set_page_config(layout="wide")
st.title("🧬 RA Exosome AI Platform - SaaS Edition")
st.caption("Batch Clinical + miRNA Analysis System")

# =====================
# CSV 批量上傳
# =====================
uploaded_files = st.file_uploader("📂 上傳 CSV 檔案 (可多選)", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_reports = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.subheader(f"📄 {uploaded_file.name} 預覽")
        st.dataframe(df.head())

        # 風險分數計算
        required_cols = ['glucose','cholesterol','crp','esr']
        for col in required_cols:
            if col not in df.columns:
                st.warning(f"缺少欄位: {col}")

        df['risk_score'] = df.apply(lambda row: sum([
            2 if 'glucose' in row and (row['glucose']<70 or row['glucose']>100) else 0,
            1 if 'cholesterol' in row and (row['cholesterol']<125 or row['cholesterol']>200) else 0,
            3 if 'crp' in row and row['crp']>3 else 0,
            2 if 'esr' in row and row['esr']>20 else 0
        ]), axis=1)
        df['risk_level'] = df['risk_score'].apply(lambda x: "High" if x>5 else "Low")

        col1, col2 = st.columns(2)
        col1.metric("平均風險分數", round(df["risk_score"].mean(),2))
        col2.metric("高風險人數", (df["risk_level"]=="High").sum())

        # Heatmap
        st.subheader("📊 Heatmap")
        plt.figure(figsize=(6,4))
        sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
        plt.savefig("heatmap.png")

        # PCA - 互動式
        st.subheader("🔹 PCA 分析 (互動)")
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols)>=2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(df[numeric_cols])
            df_pca = pd.DataFrame(components, columns=['PC1','PC2'])
            df_pca['risk_score'] = df['risk_score']
            fig = px.scatter(df_pca, x='PC1', y='PC2', color='risk_score', hover_data=['risk_score'])
            st.plotly_chart(fig)
            fig.write_image("pca.png")
        else:
            st.info("數值欄位不足，無法做 PCA")

        # Top miRNA
        mirna_cols = [c for c in df.columns if 'miRNA' in c]
        if len(mirna_cols)>0:
            st.subheader("📈 Top miRNA 條形圖")
            top_mirna = df[mirna_cols].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            top_mirna.plot(kind='bar', ax=ax)
            ax.set_ylabel("平均表達量")
            st.pyplot(fig)
            fig.savefig("top_mirna.png")

        # PDF
        st.subheader("📄 下載 PDF 報告")
        def generate_pdf(df):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(temp_file.name)
            styles = getSampleStyleSheet()
            content = [Paragraph("RA Exosome Analysis Report", styles["Title"])]
            for i,row in df.iterrows():
                text = f"{row.get('patient_id','NA')} - Risk: {row['risk_level']} (Score: {row['risk_score']:.2f})"
                content.append(Paragraph(text, styles["Normal"]))
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
            st.download_button(f"📄 下載 PDF - {uploaded_file.name}", f, f"{uploaded_file.name}_report.pdf")
