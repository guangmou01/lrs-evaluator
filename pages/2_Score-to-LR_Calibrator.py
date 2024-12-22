import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.graph_objects import Scatter
import io
import pandas as pd

# 高斯分布概率密度估计函数
def pdf(x, mean, sd):
    return (1.0 / (sd * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)

# 池化方差计算函数
def pool_variance(g1, g2):
    n1, n2 = len(g1), len(g2)
    g1_var, g2_var = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pool_var = ((n1 - 1) * g1_var + (n2 - 1) * g2_var) / (n1 + n2 - 2)
    return pool_var

# 基于原始高斯分布的 LR 校准
def raw_gaussian_calibration(score, cal_ss, cal_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    ss_sd, ds_sd = np.std(cal_ss, ddof=1), np.std(cal_ds, ddof=1)
    lr = 10**(np.log10(pdf(score, ss_mean, ss_sd)) - np.log10(pdf(score, ds_mean, ds_sd)))
    return lr, ss_mean, ds_mean, ss_sd, ds_sd

def raw_gaussian_calibration_plot(score, cal_ss, cal_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    ss_sd, ds_sd = np.std(cal_ss, ddof=1), np.std(cal_ds, ddof=1)
    x_range = np.linspace(min(min(cal_ss), min(cal_ds)) - 3, max(max(cal_ss), max(cal_ds)) + 3, 5000)
    ss_pdf, ds_pdf = pdf(x_range, ss_mean, ss_sd), pdf(x_range, ds_mean, ds_sd)

    fig = go.Figure()
    fig.update_layout(
        width=1000,
        height=500,
        xaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True),
        yaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True))

    # 添加 Evidence Score 垂直线
    fig.add_trace(go.Scatter(
        x=[score, score],
        y=[0, max(max(ss_pdf), max(ds_pdf))],
        mode='lines',
        name='Evidence Score',
        line=dict(color='green', dash='dot')))

    # 添加 Cal SS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ss,
        histnorm='probability density',
        name='SS Histogram',
        marker=dict(color='red', opacity=0.4),
        nbinsx=25))

    # 添加 Cal SS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ss_pdf,
        mode='lines',
        name='SS PDF',
        line=dict(color='red')))

    # 添加 Cal DS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ds,
        histnorm='probability density',
        name='DS Histogram',
        marker=dict(color='blue', opacity=0.4),
        nbinsx=25))

    # 添加 Cal DS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ds_pdf,
        mode='lines',
        name='DS PDF',
        line=dict(color='blue')))

    # 设置布局
    fig.update_layout(
        xaxis_title="Non-calibrated Score",
        yaxis_title="Probability Density",
        template=None,
        barmode="overlay",
        showlegend=False)

    return fig

def raw_gaussian_calibration_test(cal_ss, cal_ds, test_ss, test_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    ss_sd, ds_sd = np.std(cal_ss, ddof=1), np.std(cal_ds, ddof=1)
    test_ss_lr = 10**(np.log10(pdf(test_ss, ss_mean, ss_sd)) - np.log10(pdf(test_ss, ds_mean, ds_sd)))
    test_ds_lr = 10**(np.log10(pdf(test_ds, ss_mean, ss_sd)) - np.log10(pdf(test_ds, ds_mean, ds_sd)))
    return test_ss_lr, test_ds_lr, ss_mean, ds_mean, ss_sd, ds_sd

# 基于等方差高斯分布的 LR 校准
def equal_variance_gaussian_calibration(score, cal_ss, cal_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_sd = np.sqrt(pool_var)
    lr = 10 ** (np.log10(pdf(score, ss_mean, pool_sd)) - np.log10(pdf(score, ds_mean, pool_sd)))
    return lr, ss_mean, ds_mean, pool_sd

def equal_variance_gaussian_calibration_plot(score, cal_ss, cal_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_sd = np.sqrt(pool_var)
    x_range = np.linspace(min(min(cal_ss), min(cal_ds)) - 3, max(max(cal_ss), max(cal_ds)) + 3, 5000)
    ss_pdf = pdf(x_range, ss_mean, pool_sd)
    ds_pdf = pdf(x_range, ds_mean, pool_sd)

    fig = go.Figure()
    fig.update_layout(
        width=1000,
        height=500,
        xaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True),
        yaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True))

    # 添加 Evidence Score 垂直线
    fig.add_trace(go.Scatter(
        x=[score, score],
        y=[0, max(max(ss_pdf), max(ds_pdf))],
        mode='lines',
        name='Evidence Score',
        line=dict(color='green', dash='dot')))

    # 添加 Cal SS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ss,
        histnorm='probability density',
        name='SS Histogram',
        marker=dict(color='red', opacity=0.4),
        nbinsx=25))

    # 添加 Cal SS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ss_pdf,
        mode='lines',
        name='SS PDF',
        line=dict(color='red')))

    # 添加 Cal DS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ds,
        histnorm='probability density',
        name='DS Histogram',
        marker=dict(color='blue', opacity=0.4),
        nbinsx=25))

    # 添加 Cal DS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ds_pdf,
        mode='lines',
        name='DS PDF',
        line=dict(color='blue')))

    # 设置布局
    fig.update_layout(
        xaxis_title="Non-calibrated Score",
        yaxis_title="Probability Density",
        template=None,
        barmode="overlay",
        showlegend=False)

    return fig

def equal_variance_gaussian_calibration_test(cal_ss, cal_ds, test_ss, test_ds):
    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_sd = np.sqrt(pool_var)
    test_ss_lr = 10**(np.log10(pdf(test_ss, ss_mean, pool_sd)) - np.log10(pdf(test_ss, ds_mean, pool_sd)))
    test_ds_lr = 10**(np.log10(pdf(test_ds, ss_mean, pool_sd)) - np.log10(pdf(test_ds, ds_mean, pool_sd)))
    return test_ss_lr, test_ds_lr, ss_mean, ds_mean, pool_sd

# 基于线性逻辑回归的 LR 校准
def linear_logistic_regression_calibration(score, cal_ss, cal_ds,
                                           solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3, c=0.0001):
    scores = np.concatenate((cal_ss, cal_ds)).reshape(-1, 1)
    labels = np.array([1] * len(cal_ss) + [0] * len(cal_ds))
    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, tol=tol, C=c)
    model.fit(scores, labels)
    alpha = model.intercept_[0]
    beta = model.coef_[0][0]
    lr = 10**(alpha + beta * score)
    return lr, alpha, beta

def linear_logistic_regression_calibration_plot(score, cal_ss, cal_ds,
                                                solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3, c=0.0001):
    # Prepare the data
    scores = np.concatenate((cal_ss, cal_ds)).reshape(-1, 1)
    labels = np.array([1] * len(cal_ss) + [0] * len(cal_ds))

    # Train the logistic regression model
    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, tol=tol, C=c)
    model.fit(scores, labels)

    # Extract the coefficients
    alpha = model.intercept_[0]
    beta = model.coef_[0][0]

    # Generate a range for the x-axis (score)
    x_range = np.linspace(min(scores)[0], max(scores)[0], 5000)
    sigmoid_curve = 1 / (1 + np.exp(-(alpha + beta * x_range)))  # Logistic (sigmoid) function
    calibration_line = alpha + beta * x_range                    # Calibration line
    y_range = [min(calibration_line), max(calibration_line)]

    # Create subplots with two panels
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add sigmoid curve (top panel)
    fig.add_trace(Scatter(x=x_range, y=sigmoid_curve, mode='lines', name='Sigmoid Curve',
                          line=dict(color='green')), row=1, col=1)
    fig.add_trace(Scatter(x=cal_ss, y=[1] * len(cal_ss), mode='markers', name='Cal SS Data',
                          marker=dict(color='red', size=8, opacity=0.5)), row=1, col=1)
    fig.add_trace(Scatter(x=cal_ds, y=[0] * len(cal_ds), mode='markers', name='Cal DS Data',
                          marker=dict(color='blue', size=8, opacity=0.5)), row=1, col=1)
    fig.add_trace(Scatter(x=[score, score], y=[0, 1], mode='lines',
                          name='Evidence Score', line=dict(color='green', dash='dot')), row=1, col=1)
    fig.add_trace(Scatter(x=[min(x_range), max(x_range)], y=[0.5, 0.5], mode='lines', name='y=0.5 Line',
                          line=dict(color='black', dash='dash')), row=1, col=1)

    # Add black solid lines at y=0 and y=1
    fig.add_trace(Scatter(x=[min(x_range), max(x_range)], y=[0, 0], mode='lines', name='y=0 Line',
                          line=dict(color='black', width=1)), row=1, col=1)
    fig.add_trace(Scatter(x=[min(x_range), max(x_range)], y=[1, 1], mode='lines', name='y=1 Line',
                          line=dict(color='black', width=1)), row=1, col=1)

    # Add calibration line (bottom panel)
    fig.add_trace(Scatter(x=x_range, y=calibration_line, mode='lines', name='Calibration Line',
                          line=dict(color='green')), row=2, col=1)
    fig.add_trace(Scatter(x=[score, score], y=y_range, mode='lines',
                          name='Evidence Score', line=dict(color='green', dash='dot')), row=2, col=1)
    fig.add_trace(Scatter(x=[min(x_range), max(x_range)], y=[0, 0], mode='lines', name='y=0 Line',
                          line=dict(color='black', dash='dash')), row=2, col=1)

    # Update layout for black borders and no gaps
    fig.update_layout(
        height=800, width=800,
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, ticks="outside",
                   range=[min(x_range), max(x_range)]),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, ticks="outside"),
        xaxis2=dict(title="Non-calibrated Score", showline=True, linewidth=2, linecolor='black', mirror=True,
                    ticks="outside",
                    range=[min(x_range), max(x_range)]),
        yaxis1=dict(title="Probability", showline=True, linewidth=2, linecolor='black', mirror=True, ticks="outside"),
        yaxis2=dict(title="Calibrated log-LR", showline=True, linewidth=2, linecolor='black', mirror=True,
                    ticks="outside", range=[alpha + beta * min(x_range), alpha + beta * max(x_range)]),
        template=None,
        showlegend=False
    )

    return fig

def linear_logistic_regression_calibration_test(cal_ss, cal_ds, test_ss, test_ds,
                                                solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3, c=0.0001):
    scores = np.concatenate((cal_ss, cal_ds)).reshape(-1, 1)
    labels = np.array([1] * len(cal_ss) + [0] * len(cal_ds))
    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, tol=tol, C=c)
    model.fit(scores, labels)
    alpha = model.intercept_[0]
    beta = model.coef_[0][0]
    test_ss_lr = 10**(alpha + beta * test_ss)
    test_ds_lr = 10**(alpha + beta * test_ds)
    return test_ss_lr, test_ds_lr, alpha, beta

# 基于贝叶斯模型的 LR 校准
def bayes_calibration(score, cal_ss, cal_ds, ns, nd):
    ss_mean = np.mean(cal_ss)
    ds_mean = np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_n = (ns + nd)/2
    df = ns + nd - 2
    scaling_factor = 2/(pool_n - 1) + 1
    lr = np.exp(t.logpdf(score, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                t.logpdf(score, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    return lr, ss_mean, ds_mean, pool_var, df

def bayes_calibration_plot(score, cal_ss, cal_ds, ns, nd):
    ss_mean = np.mean(cal_ss)
    ds_mean = np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_n = (ns + nd)/2
    df = ns + nd - 2
    scaling_factor = 2/(pool_n - 1) + 1
    x_range = np.linspace(min(min(cal_ss), min(cal_ds)) - 2, max(max(cal_ss), max(cal_ds)) + 2, 5000)
    ss_pdf = np.exp(t.logpdf(x_range, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)))
    ds_pdf = np.exp(t.logpdf(x_range, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))

    fig = go.Figure()
    fig.update_layout(
        width=1000,
        height=500,
        xaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True),
        yaxis=dict(showgrid=True,
                   gridcolor="#d3d3d3",
                   gridwidth=1,
                   showline=True,
                   linecolor="black",
                   linewidth=2,
                   mirror=True))

    # 添加 Evidence Score 垂直线
    fig.add_trace(go.Scatter(
        x=[score, score],
        y=[0, max(max(ss_pdf), max(ds_pdf))],
        mode='lines',
        name='Evidence Score',
        line=dict(color='green', dash='dot')))

    # 添加 Cal SS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ss,
        histnorm='probability density',
        name='SS Histogram',
        marker=dict(color='red', opacity=0.4),
        nbinsx=25))

    # 添加 Cal SS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ss_pdf,
        mode='lines',
        name='SS PDF',
        line=dict(color='red')))

    # 添加 Cal DS 数据的直方图
    fig.add_trace(go.Histogram(
        x=cal_ds,
        histnorm='probability density',
        name='DS Histogram',
        marker=dict(color='blue', opacity=0.4),
        nbinsx=25))

    # 添加 Cal DS 高斯分布曲线
    fig.add_trace(go.Scatter(
        x=x_range,
        y=ds_pdf,
        mode='lines',
        name='DS PDF',
        line=dict(color='blue')))

    # 设置布局
    fig.update_layout(
        xaxis_title="Non-calibrated Score",
        yaxis_title="Probability Density",
        template=None,
        barmode="overlay",
        showlegend=False)

    return fig

def bayes_calibration_test(cal_ss, cal_ds, test_ss, test_ds, ns, nd):
    ss_mean = np.mean(cal_ss)
    ds_mean = np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_n = (ns + nd)/2
    df = ns + nd - 2
    scaling_factor = 2/(pool_n - 1) + 1
    test_ss_lr = np.exp(t.logpdf(test_ss, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                        t.logpdf(test_ss, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    test_ds_lr = np.exp(t.logpdf(test_ds, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                        t.logpdf(test_ds, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    return test_ss_lr, test_ds_lr, ss_mean, ds_mean, pool_var, df

# 测试结果评估
def eer(ss_lr, ds_lr):
    ss_log_lr = np.log10(ss_lr)
    ds_log_lr = np.log10(ds_lr)
    num_log_thresholds = 50000
    min_log_threshold = min(np.min(ss_log_lr), np.min(ds_log_lr))
    max_log_threshold = max(np.max(ss_log_lr), np.max(ds_log_lr))
    if min_log_threshold == -np.inf:
        min_log_threshold = np.finfo(float).tiny
    if max_log_threshold == np.inf:
        max_log_threshold = np.finfo(float).max
    log_thresholds = np.linspace(min_log_threshold, max_log_threshold, num_log_thresholds)
    ss_error = np.zeros(num_log_thresholds)
    ds_error = np.zeros(num_log_thresholds)
    for i, log_threshold in enumerate(log_thresholds):
        ss_error[i] = np.sum(ss_log_lr < log_threshold)
        ds_error[i] = np.sum(ds_log_lr > log_threshold)
    fpr = ss_error / len(ss_lr)
    fnr = ds_error / len(ds_lr)
    min_diff = np.min(np.abs(fpr - fnr))
    indexes = np.where(np.abs(fpr - fnr) == min_diff)[0]
    min_err_log_threshold = log_thresholds[indexes[0]]
    max_err_log_threshold = log_thresholds[indexes[-1]]
    mid_err_log_threshold = (min_err_log_threshold + max_err_log_threshold) / 2
    m_fpr = np.sum(ss_log_lr < mid_err_log_threshold) / len(ss_lr)
    m_fnr = np.sum(ds_log_lr > mid_err_log_threshold) / len(ds_lr)
    eer_value = (m_fpr + m_fnr) / 2
    eer_threshold = 10 ** mid_err_log_threshold
    return eer_value, eer_threshold

def cllr(ss_lr, ds_lr):
    punish_ss = np.log2(1 + (1 / ss_lr))
    punish_ds = np.log2(1 + ds_lr)
    n_vali_ss = len(ss_lr)
    n_vali_ds = len(ds_lr)
    cllr_value = 0.5 * (1 / n_vali_ss * sum(punish_ss) + 1 / n_vali_ds * sum(punish_ds))
    return cllr_value

def tippett_plot(ss_lr, ds_lr, evidence_lr, line_type):
    # Ensure input is 1D numpy arrays
    ss_lr = np.asarray(ss_lr).flatten()
    ds_lr = np.asarray(ds_lr).flatten()

    # Handle empty arrays
    if ss_lr.size == 0 or ds_lr.size == 0:
        raise ValueError("Input arrays for ss_lr and ds_lr must not be empty.")

    # Sort SS and DS likelihood ratios
    ss_lr_sorted = np.sort(np.log10(ss_lr))
    ss_cumulative = np.arange(1, len(ss_lr_sorted) + 1) / len(ss_lr_sorted)
    ds_lr_sorted = np.sort(np.log10(ds_lr))[::-1]
    ds_cumulative = np.arange(1, len(ds_lr_sorted) + 1) / len(ds_lr_sorted)

    # Determine x-axis range based on data
    x_min = np.min(np.concatenate([ss_lr_sorted, ds_lr_sorted]))
    x_max = np.max(np.concatenate([ss_lr_sorted, ds_lr_sorted]))

    # Create figure
    fig = go.Figure()

    # Add DS curve
    fig.add_trace(go.Scatter(
        x=ds_lr_sorted,
        y=ds_cumulative,
        mode='lines',
        name="FAR",
        line=dict(color='blue', dash=line_type)
    ))

    # Add SS curve
    fig.add_trace(go.Scatter(
        x=ss_lr_sorted,
        y=ss_cumulative,
        mode='lines',
        name="FRR",
        line=dict(color='red', dash=line_type)
    ))

    # Add Evidence LR vertical line
    if evidence_lr != "None":
        evidence_lr = float(evidence_lr)
        fig.add_trace(go.Scatter(
            x=[np.log10(evidence_lr), np.log10(evidence_lr)],
            y=[0, 1],
            mode='lines',
            name=f'Evidence LR: {evidence_lr}',
            line=dict(color='green', dash='dot')
        ))

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title="Log10 Likelihood Ratio",
            range=[x_min, x_max],
            showgrid=True,
            gridcolor="#d3d3d3",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            title_standoff=10
        ),
        yaxis=dict(
            title="Cumulative Proportion",
            range=[0, 1],
            tickvals=np.arange(0, 1.01, 0.2),
            showgrid=True,
            gridcolor="#d3d3d3",
            zeroline=False,
            zerolinecolor="black",
            zerolinewidth=2,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True
        ),
        showlegend=False,
        template=None,
        height=600,
        width=800
    )

    return fig

def display_and_download_stats(title, data, file_name):
    """显示 DataFrame 并提供下载功能"""
    st.write(title)
    st.dataframe(data)
    st.download_button(
        f"💾 Download as CSV",
        data.to_csv(index=False),
        file_name,
        "text/csv")

def display_and_download_lr(title, lr_values, file_name):
    """显示和下载 LR 结果"""
    with st.expander(title):
        lr_str = ", ".join([f"{val}" for val in lr_values])
        st.code(lr_str, language="plaintext")
        st.download_button(
            f"💾 Download as TXT",
            lr_str,
            file_name,
            "text/plain")

# 定义 Streamlit 应用
def main():
    st.set_page_config(page_title="LRs Evaluator",
                       page_icon="⚖️")
    st.title("⚖️ Score-to-LR Calibrator")
    st.write("Author: Guangmou"
             "  \n E-mail: forensicstats@hotmail.com")
    st.markdown("---")
    st.sidebar.title("⚙️ Setting")

    mode = st.sidebar.radio("Choose a Mode", ["Calibration Mode", "Test Mode"])
    method = st.sidebar.selectbox("Choose a Calibration Method:", [
        "Raw Gaussian Calibration",
        "equal-Variance Gaussian Calibration",
        "Logistic Regression Calibration",
        "Bayes Model Calibration"
    ])

    st.sidebar.subheader("️📏 Calibration Sets Input")
    cal_ss = st.sidebar.text_area("Input Same-source-pairs Score (log10-LR) Set:", "1.0, 1.2, 0.9, 1.1")
    cal_ds = st.sidebar.text_area("Input Different-source-pairs Score (log10-LR) Set:", "0.4, 0.5, 0.3, 0.6")
    cal_ss = np.array([float(x) for x in cal_ss.split(",")])
    cal_ds = np.array([float(x) for x in cal_ds.split(",")])

    st.sidebar.subheader("🔍️ Evidence Score Input")
    score = st.sidebar.text_area("Input a Evidence Score (log10-LR):", "1.0")
    try:
        score = float(score)
    except ValueError:
        st.error("Please enter a valid number for the evidential score.")

    st.sidebar.subheader("⚖️ Test Sets Input")
    test_ss = st.sidebar.text_area("Input Same-source-pairs Score (log10-LR) Set:", "1.0, 1.3, 0.8")
    test_ds = st.sidebar.text_area("Input Different-source-pairs Score (log10-LR) Set:", "0.2, 0.4, 0.5")
    test_ss = np.array([float(x) for x in test_ss.split(",")])
    test_ds = np.array([float(x) for x in test_ds.split(",")])

    if mode == "Calibration Mode":
        st.header("🔍 Result of Calibration")
        if method == "Raw Gaussian Calibration":
            lr, ss_mean, ds_mean, ss_sd, ds_sd = raw_gaussian_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "SS SD", "DS SD"],
                "Value": [ss_mean, ds_mean, ss_sd, ds_sd]})
            graphic_re = raw_gaussian_calibration_plot(score, cal_ss, cal_ds)

        elif method == "equal-Variance Gaussian Calibration":
            lr, ss_mean, ds_mean, pool_sd = equal_variance_gaussian_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "Pool SD"],
                "Value": [ss_mean, ds_mean, pool_sd]})
            graphic_re = equal_variance_gaussian_calibration_plot(score, cal_ss, cal_ds)

        elif method == "Logistic Regression Calibration":
            c_input = st.text_input('Input the Degree of Regularization:', value="100")
            if c_input.isdigit():
                c_value = 1/int(c_input)
                if c_value > 0:
                    lr, alpha, beta = linear_logistic_regression_calibration(score, cal_ss, cal_ds, c=c_value)
                    calibration_stats = pd.DataFrame({
                        "Metric": ["Alpha", "Beta"],
                        "Value": [alpha, beta]})
                    graphic_re = linear_logistic_regression_calibration_plot(score, cal_ss, cal_ds, c=c_value)
                else:
                    st.error("The degree of regularization should be a positive number.")
            else:
                st.warning("Please input a valid number for the degree of regularization (>1).")

        elif method == "Bayes Model Calibration":
            ns_input = st.text_input('Input the Number of Individuals in SS-Calibration Set:', value="5")
            nd_input = st.text_input('Input the Number of Individuals in DS-Calibration Set:', value="5")
            if ns_input.isdigit() and nd_input.isdigit():
                ns_value = int(ns_input)
                nd_value = int(nd_input)
                if ns_value > 1 and nd_value > 1:
                    lr, ss_mean, ds_mean, pool_var, df = bayes_calibration(
                        score, cal_ss, cal_ds, ns_value, nd_value)
                    calibration_stats = pd.DataFrame({
                        "Metric": ["SS Mean", "DS Mean", "Pool Variance", "Degrees of Freedom"],
                        "Value": [ss_mean, ds_mean, pool_var, df]})
                    graphic_re = bayes_calibration_plot(score, cal_ss, cal_ds, ns_value, nd_value)
                else:
                    st.error("Number of speakers must be greater than 1.")
            else:
                st.warning("Please input a valid number for the number of speakers.")

        # 显示校准后的 LR
        st.write(f"🧮 Non-calibrated Score (Evidence Score): {score}")
        st.write(f"🧮 Calibrated Likelihood Ratio (Evidence LR): {lr}")

        # 显示和下载 Calibration Stats
        display_and_download_stats("📊 Calibration Statistics", calibration_stats, "calibration_stats.csv")

        # 显示 Graphic Representation
        st.markdown("### 📊 Graphic Representation")
        st.plotly_chart(graphic_re, use_container_width=True)
        # Save the Figure into BytesIO for Downloading
        buf = io.BytesIO()
        graphic_re.write_image(buf, format='png', scale=4)
        buf.seek(0)

        # Download Button
        st.download_button("💾  Download the Graphic Representation", buf, "graphic_representation.png", "image/png")

    elif mode == "Test Mode":
        st.header("⚖️ Result of Test")
        # Initialize test LR variables to avoid UnboundLocalError
        test_ss_lr = np.array([])
        test_ds_lr = np.array([])
        calibration_stats = pd.DataFrame()

        if method == "Raw Gaussian Calibration":
            test_ss_lr, test_ds_lr, ss_mean, ds_mean, ss_sd, ds_sd = raw_gaussian_calibration_test(
                cal_ss, cal_ds, test_ss, test_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "SS SD", "DS SD"],
                "Value": [ss_mean, ds_mean, ss_sd, ds_sd]})

        elif method == "equal-Variance Gaussian Calibration":
            test_ss_lr, test_ds_lr, ss_mean, ds_mean, pool_sd = equal_variance_gaussian_calibration_test(
                cal_ss, cal_ds, test_ss, test_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "Pool SD"],
                "Value": [ss_mean, ds_mean, pool_sd]})

        elif method == "Logistic Regression Calibration":
            c_input = st.text_input('Input the Degree of Regularization:', value="100")
            if c_input.isdigit():
                c_value = 1/int(c_input)
                if c_value > 0:
                    test_ss_lr, test_ds_lr, alpha, beta = linear_logistic_regression_calibration_test(
                        cal_ss, cal_ds, test_ss, test_ds,
                        solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3, c=c_value)
                    calibration_stats = pd.DataFrame({
                        "Metric": ["Alpha", "Beta"],
                        "Value": [alpha, beta]})
                else:
                    st.error("The degree of regularization should be a positive number.")
            else:
                st.warning("Please input a valid number for the degree of regularization (>1).")

        elif method == "Bayes Model Calibration":
            ns_input = st.text_input('Input the Number of Individuals in SS-Calibration Set:', value="5")
            nd_input = st.text_input('Input the Number of Individuals in DS-Calibration Set:', value="5")
            if ns_input.isdigit() and nd_input.isdigit():
                ns_value = int(ns_input)
                nd_value = int(nd_input)
                if ns_value > 1 and nd_value > 1:
                    test_ss_lr, test_ds_lr, ss_mean, ds_mean, pool_var, df = bayes_calibration_test(
                        cal_ss, cal_ds, test_ss, test_ds, ns_value, nd_value)
                    calibration_stats = pd.DataFrame({
                        "Metric": ["SS Mean", "DS Mean", "Pool Variance", "Degrees of Freedom"],
                        "Value": [ss_mean, ds_mean, pool_var, df]})
                else:
                    st.error("Number of speakers must be greater than 1.")
            else:
                st.warning("Please input a valid number for the number of speakers.")

        # Compute and display Cllr, EER, and Tippett Plot only if LRs are available
        if test_ss_lr.size > 0 and test_ds_lr.size > 0:
            cllr_raw = cllr(10**test_ss, 10**test_ds)
            eer_raw, eer_threshold_raw = eer(10**test_ss, 10**test_ds)
            log10_eer_threshold_raw = np.log10(eer_threshold_raw)
            cllr_calibrated = cllr(test_ss_lr, test_ds_lr)
            eer_calibrated, eer_threshold_calibrated = eer(test_ss_lr, test_ds_lr)
            log10_eer_threshold_calibrated = np.log10(eer_threshold_calibrated)
            evaluation_stats = pd.DataFrame({
                "Metrics": ["Cllr", "EER", "EER-threshold", "Log10-EER-threshold"],
                "Non-calibrated": [cllr_raw, eer_raw, eer_threshold_raw, log10_eer_threshold_raw],
                "Calibrated": [cllr_calibrated, eer_calibrated, eer_threshold_calibrated, log10_eer_threshold_calibrated]})

            # Display and download results
            display_and_download_lr("📋 Calibrated Test SS LR", test_ss_lr, "test_ss_lr.txt")
            display_and_download_lr("📋 Calibrated Test DS LR", test_ds_lr, "test_ds_lr.txt")
            display_and_download_stats("📊 Calibration Statistics", calibration_stats, "calibration_stats.csv")
            display_and_download_stats("📊 Evaluation Metrics", evaluation_stats, "evaluation_metrics.csv")
        else:
            st.warning("Test LRs could not be computed. Please check your input values.")

        # Tippett Plot Optional Settings
        with st.expander('⚙️  Tippett Setting'):
            evi_value_input = st.text_input('Input the Evidence LR ("None" or input a valid positive number)',
                                            'None')
            line_type_input = st.selectbox('Line Type', ['solid', 'dotted', 'dashed', 'dash-dot'])

            # Generate the Tippett Plot
            if st.button("📈  Generate the Tippett Plot", key="tippett_button"):
                # Map valid line types for Plotly
                line_type_mapping = {
                    'solid': 'solid',
                    'dotted': 'dot',
                    'dashed': 'dash',
                    'dash-dot': 'dashdot'
                }
                line_type_input = line_type_mapping.get(line_type_input, 'solid')  # Default to 'solid'
                try:
                    tippett_fig = tippett_plot(test_ss_lr, test_ds_lr, evi_value_input,
                                               line_type_input)
                    st.plotly_chart(tippett_fig, use_container_width=True)

                    # Save the Figure into BytesIO for Downloading
                    buf = io.BytesIO()
                    tippett_fig.write_image(buf, format='png', scale=4)
                    buf.seek(0)

                    # Download Button
                    st.download_button("💾  Download the Tippett Plot", buf, "tippett_plot.png", "image/png")
                except Exception as e:
                    st.error(f"Error generating tippett plot: {str(e)}")

if __name__ == "__main__":
    main()
