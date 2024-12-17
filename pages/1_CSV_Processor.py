import streamlit as st
import pandas as pd
import io
import numpy as np

# 页面设置
st.set_page_config(page_title="LRs Evaluator",
                   page_icon="⚖️")
st.title("📃️ CSV Processor")
st.write("Author: Guangmou"
         "  \n E-mail: forensicstats@hotmail.com")
st.markdown("---")

# 文件上传
st.markdown("### Please Upload a .csv File")
uploaded_file = st.file_uploader("Upload a target-file:", type=["csv"])

if uploaded_file is not None:
    # 用户选择是否有标题行
    st.markdown("### Does It Have a Header?")
    header_option = st.radio("Choose whether to remove the header:", options=["Yes", "No"])

    # 读取数据
    if header_option == "Yes":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, header=None)

    # 展示数据
    st.markdown("### Dataframe:")
    st.dataframe(df)

    if not df.empty:
        # 列选择
        col_options = df.columns if header_option == "Yes" else [f"Column {i+1}" for i in range(df.shape[1])]
        selected_col = st.selectbox("Which column do you want to extract:", options=col_options)

        # 处理所选列
        if header_option == "Yes":
            column_data = df[selected_col].dropna()
        else:
            col_index = int(selected_col.split(" ")[1]) - 1
            column_data = df.iloc[:, col_index].dropna()

        # 添加数据转换选项
        st.markdown("### Transformation Options:")
        transformation = st.radio("Choose a transformation:",
                                  options=["➡️ Raw", "➡️ Base-10 Logarithm", "➡️ Power of 10"])

        # 转换处理
        try:
            if transformation == "➡️ Base-10 Logarithm":
                column_data = column_data.astype(float)
                transformed_data = np.log10(column_data.replace(0, np.nan)).dropna()
                st.warning("⚠️ Values <= 0 are removed during Log10 transformation.")
            elif transformation == "➡️ Power of 10":
                column_data = column_data.astype(float)
                transformed_data = 10 ** column_data
            else:
                transformed_data = column_data.astype(str)

            # 将数据转换为字符串向量
            vector_data = ",".join(transformed_data.astype(str))

            # 展示预览数据
            st.markdown("### Preview:")
            st.code(vector_data)

            # 创建下载文件
            buffer = io.BytesIO()
            buffer.write(vector_data.encode("utf-8"))
            buffer.seek(0)

            st.download_button(
                label="💾  Download the .txt file",
                data=buffer,
                file_name=f"{selected_col}.txt",
                mime="text/plain"
            )
        except ValueError:
            st.error("❌ Error: Ensure the selected column contains only numeric values for transformation.")