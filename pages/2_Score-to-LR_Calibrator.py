import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
import io
import pandas as pd

# 高斯分布概率密度估计函数
def pdf(x, mean, sd):
    return (1.0 / (sd * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)

# 池化方差计算函数
def pool_variance(g1, g2):

    n1, n2 = len(g1), len(g2)
    g1_var = np.var(g1, ddof=1)
    g2_var = np.var(g2, ddof=1)

    pool_var = ((n1 - 1) * g1_var + (n2 - 1) * g2_var) / (n1 + n2 - 2)
    return pool_var

# 基于原始高斯分布的 LR 校准
def raw_gaussian_calibration(score, cal_ss, cal_ds):

    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    ss_sd, ds_sd = np.std(cal_ss, ddof=1), np.std(cal_ds, ddof=1)

    lr = 10**(np.log10(pdf(score, ss_mean, ss_sd)) - np.log10(pdf(score, ds_mean, ds_sd)))
    return lr, ss_mean, ds_mean, ss_sd, ds_sd

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

def equal_variance_gaussian_calibration_test(cal_ss, cal_ds, test_ss, test_ds):

    ss_mean, ds_mean = np.mean(cal_ss), np.mean(cal_ds)
    pool_var = pool_variance(cal_ss, cal_ds)
    pool_sd = np.sqrt(pool_var)

    test_ss_lr = 10**(np.log10(pdf(test_ss, ss_mean, pool_sd)) - np.log10(pdf(test_ss, ds_mean, pool_sd)))
    test_ds_lr = 10**(np.log10(pdf(test_ds, ss_mean, pool_sd)) - np.log10(pdf(test_ds, ds_mean, pool_sd)))

    return test_ss_lr, test_ds_lr, ss_mean, ds_mean, pool_sd

# 基于线性逻辑回归的 LR 校准
def linear_logistic_regression_calibration(score, cal_ss, cal_ds,
                                           solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3):

    scores = np.concatenate((cal_ss, cal_ds)).reshape(-1, 1)
    labels = np.array([1] * len(cal_ss) + [0] * len(cal_ds))

    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, tol=tol)
    model.fit(scores, labels)

    alpha = model.intercept_[0]
    beta = model.coef_[0][0]

    lr = 10**(alpha + beta * score)
    return lr, alpha, beta

def linear_logistic_regression_calibration_test(cal_ss, cal_ds, test_ss, test_ds,
                                                solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3):

    scores = np.concatenate((cal_ss, cal_ds)).reshape(-1, 1)
    labels = np.array([1] * len(cal_ss) + [0] * len(cal_ds))

    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, tol=tol)
    model.fit(scores, labels)

    alpha = model.intercept_[0]
    beta = model.coef_[0][0]

    test_ss_lr = 10**(alpha + beta * test_ss)
    test_ds_lr = 10**(alpha + beta * test_ds)

    return test_ss_lr, test_ds_lr, alpha, beta

# 基于贝叶斯模型的 LR 校准
def bayes_calibration(score, cal_ss, cal_ds):

    ss_mean = np.mean(cal_ss)
    ss_n = len(cal_ss)
    ds_mean = np.mean(cal_ds)
    ds_n = len(cal_ds)
    pool_n = ss_n + ds_n
    pool_var = pool_variance(cal_ss, cal_ds)

    df = ss_n + ds_n - 2
    scaling_factor = 2/(pool_n - 1) + 1

    lr = np.exp(t.logpdf(score, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                t.logpdf(score, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    return lr, ss_mean, ds_mean, pool_var, df

def bayes_calibration_test(cal_ss, cal_ds, test_ss, test_ds):
    ss_mean = np.mean(cal_ss)
    ss_n = len(cal_ss)
    ds_mean = np.mean(cal_ds)
    ds_n = len(cal_ds)
    pool_n = ss_n + ds_n
    pool_var = pool_variance(cal_ss, cal_ds)

    df = ss_n + ds_n - 2
    scaling_factor = 2/(pool_n - 1) + 1

    test_ss_lr = np.exp(t.logpdf(test_ss, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                        t.logpdf(test_ss, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    test_ds_lr = np.exp(t.logpdf(test_ds, df, loc=ss_mean, scale=np.sqrt(scaling_factor * pool_var)) -
                        t.logpdf(test_ds, df, loc=ds_mean, scale=np.sqrt(scaling_factor * pool_var)))
    return test_ss_lr, test_ds_lr, ds_mean, pool_var, df

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

def tippett_plot(ss_lr, ds_lr, evidence_lr,
                 x_range, y_range,
                 ss_lr_tag, ds_lr_tag,
                 line_type,
                 legend_pos,
                 e_pos):
    ss_lr_sorted = np.sort(np.log10(ss_lr))
    ss_cumulative = np.arange(1, len(ss_lr_sorted) + 1) / len(ss_lr_sorted)
    ds_lr_sorted = np.sort(np.log10(ds_lr))[::-1]
    ds_cumulative = np.arange(1, len(ds_lr_sorted) + 1) / len(ds_lr_sorted)

    fig_tippett, ax = plt.subplots(figsize=(8, 6))
    # Draw the Tippett Plot
    ax.plot(ds_lr_sorted, ds_cumulative, label=ds_lr_tag, color='blue', linestyle=line_type)
    ax.plot(ss_lr_sorted, ss_cumulative, label=ss_lr_tag, color='red', linestyle=line_type)
    ax.axvline(0, color='black', linestyle='--')
    if legend_pos != 'None':
        ax.legend(loc=legend_pos)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Log10 Likelihood Ratio')
    ax.set_ylabel('Cumulative Proportion')
    ax.grid(True, alpha=0.4)
    if evidence_lr != "None":
        evidence_lr = float(evidence_lr)
        ax.axvline(np.log10(evidence_lr), color='green', linestyle='-', alpha=0.6)  # Draw the Evidence Line
        ax.annotate(f'E = {evidence_lr}',
                    xy=(np.log10(evidence_lr), 0.5),
                    xytext=(np.log10(evidence_lr)+(max(x_range)-min(x_range))/50, 0.5),
                    fontsize=10, ha=e_pos, color='black')
        # Annotate the Evidence Value

    return fig_tippett

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
    method = st.sidebar.selectbox("Choose a Calibration Method", [
        "Raw Gaussian Calibration",
        "equal-Variance Gaussian Calibration",
        "Logistic Regression Calibration",
        "Bayes Model Calibration"
    ])

    st.sidebar.subheader("️📏 Calibration Sets Input")
    cal_ss = st.sidebar.text_area("Input Same-source-pairs Score Set:", "1.0, 1.2, 0.9, 1.1")
    cal_ds = st.sidebar.text_area("Input Different-source-pairs Score Set", "0.4, 0.5, 0.3, 0.6")
    cal_ss = np.array([float(x) for x in cal_ss.split(",")])
    cal_ds = np.array([float(x) for x in cal_ds.split(",")])

    st.sidebar.subheader("🔍️ Evidential Score Input")
    score = st.sidebar.text_area("Input a Evidential Score:", "1.0")
    try:
        score = float(score)
    except ValueError:
        st.error("Please enter a valid number for the evidential score.")

    st.sidebar.subheader("⚖️ Test Sets Input")
    test_ss = st.sidebar.text_area("Input Same-source-pairs Score Set:", "1.0, 1.3, 0.8")
    test_ds = st.sidebar.text_area("Input Different-source-pairs Score Set:", "0.2, 0.4, 0.5")
    test_ss = np.array([float(x) for x in test_ss.split(",")])
    test_ds = np.array([float(x) for x in test_ds.split(",")])

    if mode == "Calibration Mode":
        st.header("🔍 Result of Calibration")
        if method == "Raw Gaussian Calibration":
            lr, ss_mean, ds_mean, ss_sd, ds_sd = raw_gaussian_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "SS SD", "DS SD"],
                "Value": [ss_mean, ds_mean, ss_sd, ds_sd]})

        elif method == "equal-Variance Gaussian Calibration":
            lr, ss_mean, ds_mean, pool_sd = equal_variance_gaussian_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "Pool SD"],
                "Value": [ss_mean, ds_mean, pool_sd]})

        elif method == "Logistic Regression Calibration":
            lr, alpha, beta = linear_logistic_regression_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["Alpha", "Beta"],
                "Value": [alpha, beta]})

        elif method == "Bayes Model Calibration":
            lr, ss_mean, ds_mean, pool_var, df = bayes_calibration(score, cal_ss, cal_ds)
            calibration_stats = pd.DataFrame({
                "Metric": ["SS Mean", "DS Mean", "Pool Variance", "Degrees of Freedom"],
                "Value": [ss_mean, ds_mean, pool_var, df]})

        # 显示校准后的 LR
        st.write(f"🧮 non-Calibrated Score (Evidence Score): {score}")
        st.write(f"🧮 Calibrated Likelihood Ratio (Evidence LR): {lr}")

        # 显示和下载 Calibration Stats
        display_and_download_stats("📊 Calibration Statistics", calibration_stats, "calibration_stats.csv")

    elif mode == "Test Mode":
        st.header("⚖️ Result of Test")
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
            test_ss_lr, test_ds_lr, alpha, beta = linear_logistic_regression_calibration_test(
                cal_ss, cal_ds, test_ss, test_ds,
                solver='liblinear', penalty="l2", max_iter=5000, tol=1e-3)

            calibration_stats = pd.DataFrame({
                "Metric": ["Alpha", "Beta"],
                "Value": [alpha, beta]})

        elif method == "Bayes Model Calibration":
            test_ss_lr, test_ds_lr, ds_mean, pool_var, df = bayes_calibration_test(
                cal_ss, cal_ds, test_ss, test_ds)

            calibration_stats = pd.DataFrame({
                "Metric": ["DS Mean", "Pool Variance", "Degrees of Freedom"],
                "Value": [ds_mean, pool_var, df]})

        # 计算 CLLR 和 EER
        cllr_value = cllr(test_ss_lr, test_ds_lr)
        eer_value, eer_threshold = eer(test_ss_lr, test_ds_lr)
        log10_eer_threshold = np.log10(eer_threshold)

        evaluation_stats = pd.DataFrame({
            "Metric": ["Cllr", "EER", "EER-threshold", "Log10-EER-threshold"],
            "Value": [cllr_value, eer_value, eer_threshold, log10_eer_threshold]})

        # 显示和下载 Test SS 和 DS LR
        display_and_download_lr("📋 Calibrated Test SS LR", test_ss_lr, "test_ss_lr.txt")
        display_and_download_lr("📋 Calibrated Test DS LR", test_ds_lr, "test_ds_lr.txt")

        # 显示和下载 Calibration Stats
        display_and_download_stats("📊 Calibration Statistics", calibration_stats, "calibration_stats.csv")

        # 显示和下载 Evaluation Metrics
        display_and_download_stats("📊 Evaluation Metrics", evaluation_stats, "evaluation_metrics.csv")

        # Tippett Plot Optional Settings
        with st.expander('⚙️  Tippett Setting'):
            evi_value_input = st.text_input('Input the Evidence LR ("None" or input a valid positive number)',
                                            'None')
            col1, col2 = st.columns(2)
            with col1:
                x_range_input_tippett = st.text_input('X Range (e.g. -5, 5)', '-5, 5', key="x_range_tippett")
            with col2:
                y_range_input_tippett = st.text_input('Y Range (e.g. 0, 1)', '0, 1', key="y_range_tippett")
            col1, col2 = st.columns(2)
            with col1:
                ss_lr_tag_input = st.text_input('Name the Tag for Positive Pairs', 'Same Source')
            with col2:
                ds_lr_tag_input = st.text_input('Name the Tag for Negative Pairs', 'Different Source')
            line_type_input = st.selectbox('Line Type', ['solid', 'dotted', 'dashed', 'dash-dot'])
            col1, col2 = st.columns(2)
            with col1:
                legend_pos_input = st.selectbox('Position of Legend', ['None',
                                                                       'lower left', 'lower right',
                                                                       'center left', 'center right',
                                                                       'upper left', 'upper right'])
            with col2:
                e_pos_input = st.selectbox('Position of Evidence LR', ["right", "left"])
                if e_pos_input == "left":
                    e_pos_input = "right"
                else:
                    e_pos_input = "left"
            try:
                x_range_input_tippett = np.array([float(x) for x in x_range_input_tippett.split(',')])
                y_range_input_tippett = np.array([float(x) for x in y_range_input_tippett.split(',')])
            except ValueError:
                st.error("Please enter valid numbers, separated by commas.")

            # Generate the Tippett Plot
            if st.button("📈  Generate the Tippett Plot", key="tippett_button"):
                # Convert the Line Type
                if line_type_input == 'solid':
                    line_type_input = '-'
                elif line_type_input == 'dotted':
                    line_type_input = ':'
                elif line_type_input == 'dashed':
                    line_type_input = '--'
                elif line_type_input == 'dash-dot':
                    line_type_input = '-.'
                try:
                    tippett_fig = tippett_plot(test_ss_lr, test_ds_lr, evi_value_input,
                                               x_range_input_tippett, y_range_input_tippett,
                                               ss_lr_tag_input, ds_lr_tag_input,
                                               line_type_input,
                                               legend_pos_input, e_pos_input)
                    st.pyplot(tippett_fig)

                    # Save the Figure into BytesIO for Downloading
                    buf = io.BytesIO()
                    tippett_fig.savefig(buf, format='png', dpi=600)
                    buf.seek(0)

                    # Download Button
                    st.download_button("💾  Download the Tippett Plot", buf, "tippett_plot.png", "image/png")
                except Exception as e:
                    st.error(f"Error generating tippett plot: {str(e)}")


if __name__ == "__main__":
    main()