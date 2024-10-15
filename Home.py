import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# Header
st.set_page_config(page_title="LRs Evaluator",
                   page_icon="⚖️")
st.title("⚖️ LRs Evaluator")
st.subheader("A Tookit for Forensic LR-based System Evaluation, With Friendly UI Workspace.")
st.write("Author: Guangmou"
         "  \n E-mail: forensicstats@hotmail.com")
st.markdown("---")

# pip install numpy
# pip install matplotlib
# pip install scikit-learn
# pip install pandas
# streamlit run Home.py
# pip freeze > requirements.txt