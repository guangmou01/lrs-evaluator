import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import io

def tippett_plot(ss_lr, ds_lr, evidence_lr,
                 x_range, y_range,
                 ss_lr_tag, ds_lr_tag,
                 line_type,
                 legend_pos):
    fig_tippett, ax = plt.subplots(figsize=(8, 6))
    ss_lr_sorted = np.sort(np.log10(ss_lr))
    ss_cumulative = np.arange(1, len(ss_lr_sorted) + 1) / len(ss_lr_sorted)
    ds_lr_sorted = np.sort(np.log10(ds_lr))[::-1]
    ds_cumulative = np.arange(1, len(ds_lr_sorted) + 1) / len(ds_lr_sorted)

    ax.plot(ds_lr_sorted, ds_cumulative, label=ds_lr_tag, color='blue', linestyle=line_type)
    ax.plot(ss_lr_sorted, ss_cumulative, label=ss_lr_tag, color='red', linestyle=line_type)
    ax.axvline(0, color='black', linestyle='--')
    ax.legend(loc=legend_pos)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Log10 Likelihood Ratio')
    ax.set_ylabel('Cumulative Proportion')
    ax.grid(True, alpha=0.4)

    if evidence_lr != "None":
        evidence_lr = float(evidence_lr)
        ax.axvline(np.log10(evidence_lr), color='green', linestyle='-', alpha=0.6)
        ax.annotate(f'E = {evidence_lr}',
                    xy=(np.log10(evidence_lr), 0.5),
                    xytext=(np.log10(evidence_lr)+(max(x_range)-min(x_range))/50, 0.5),
                    fontsize=10, ha='left', color='black')

    return fig_tippett

def plot_roc_curve(ss_lr, ds_lr, x_range, y_range, show_auc):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    auc_value = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    fig_roc, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label='ROC Curve', color='red')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
    if show_auc == "Yes":
        ax.annotate(f'AUC = {auc_value:.4f}',
                    xy=(0.61, 0.45),
                    fontsize=10,
                    color='black')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc='lower right')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.4)

    return fig_roc

def plot_det_curve(ss_lr, ds_lr, eer_value, x_range, y_range, show_eer_point):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fig_det, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, fnr, label='DET Curve', color='blue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
    if show_eer_point == "Yes":
        ax.scatter(eer_value, eer_value, s=20, color='red', label="EER Point", marker='o', zorder=8)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('False Negative Rate (FNR)')
    ax.legend(loc='upper right')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.4)

    return fig_det

def auc(ss_lr, ds_lr):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    auc_value = roc_auc_score(labels, scores)

    return auc_value

def eer(ss_lr, ds_lr):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_value = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    return eer_value, eer_threshold

def cllr(ss_lr, ds_lr):
    punish_ss = np.log2(1 + (1 / ss_lr))
    punish_ds = np.log2(1 + ds_lr)
    n_vali_ss = len(ss_lr)
    n_vali_ds = len(ds_lr)
    cllr_value = 0.5 * (1 / n_vali_ss * sum(punish_ss) + 1 / n_vali_ds * sum(punish_ds))

    return cllr_value


st.set_page_config(page_title="LR-based System Evaluator",
                   page_icon="⚖️")
st.title("⚖️ LR-based System Evaluator")

# Get Input LRs and Other Parameters
ss_lr_input = st.text_input('LR Values of Positive Pairs',
                             '0.8, 1, 5, 3, 9, 10, 25, 6, 20, 18, 0.9, 0.7, 11')
ds_lr_input = st.text_input('LR Values of Negative Pairs',
                             '0.002, 0.01, 0.3, 0.5, 0.9, 1.2, 0.6, 0.05, 0.006, 1.3, 0.4, 0.2, 0.03, 1.1, 2, 0.0005')

with st.expander('⚙️  ROC Setting'):
    col1, col2 = st.columns(2)
    with col1:
        x_range_input_roc = st.text_input('X Range (e.g. 0, 1)', '0, 1', key="x_range_roc")
    with col2:
        y_range_input_roc = st.text_input('Y Range (e.g. 0, 1)', '0, 1', key="y_range_roc")
    show_auc_input = st.selectbox('Show the AUC', ['Yes', 'No'])

with st.expander('⚙️  DET Setting'):
    col1, col2 = st.columns(2)
    with col1:
        x_range_input_det = st.text_input('X Range (e.g. 0, 1)', '0, 1', key="x_range_det")
    with col2:
        y_range_input_det = st.text_input('Y Range (e.g. 0, 1)', '0, 1', key="y_range_det")
    show_eer_point_input = st.selectbox('Show the EER point', ['Yes', 'No'])

with st.expander('⚙️  Tippett Setting'):
    evi_value_input = st.text_input('Input the Evidence LR ("None" or input a valid positive number)',
                                    'None')
    col1, col2 = st.columns(2)
    with col1:
        x_range_input_tippett = st.text_input('X Range (e.g. -3.5, 1.5)', '-3.5, 1.5', key="x_range_tippett")
    with col2:
        y_range_input_tippett = st.text_input('Y Range (e.g. 0, 1)', '0, 1', key="y_range_tippett")
    col1, col2 = st.columns(2)
    with col1:
        ss_lr_tag_input = st.text_input('Name the Tag for Positive Pairs', 'Same Source')
    with col2:
        ds_lr_tag_input = st.text_input('Name the Tag for Negative Pairs', 'Different Source')
    line_type_input = st.selectbox('Line Type', ['solid', 'dotted', 'dashed', 'dash-dot'])
    legend_pos_input = st.selectbox('Position of Legend', ['lower left', 'lower right',
                                                           'center left', 'center right',
                                                           'upper left', 'upper right'])

# Convert the Input String to a Floating Point Array
try:
    ss_lr_input = np.array([float(x) for x in ss_lr_input.split(',')])
    ds_lr_input = np.array([float(x) for x in ds_lr_input.split(',')])
    x_range_input_tippett = np.array([float(x) for x in x_range_input_tippett.split(',')])
    y_range_input_tippett = np.array([float(x) for x in y_range_input_tippett.split(',')])
    x_range_input_det = np.array([float(x) for x in x_range_input_det.split(',')])
    y_range_input_det = np.array([float(x) for x in y_range_input_det.split(',')])
    x_range_input_roc = np.array([float(x) for x in x_range_input_roc.split(',')])
    y_range_input_roc = np.array([float(x) for x in y_range_input_roc.split(',')])
except ValueError:
    st.error("Please enter valid numbers, separated by commas.")

# Convert the Line Type
if line_type_input == 'solid':
    line_type_input = '-'
elif line_type_input == 'dotted':
    line_type_input = ':'
elif line_type_input == 'dashed':
    line_type_input = '--'
elif line_type_input == 'dash-dot':
    line_type_input = '-.'

# Calculate the AUC
if st.button("🧮  Calculate the AUC", key="auc_button"):
    try:
        auc = auc(ss_lr_input, ds_lr_input)
        st.write(f"⚙️  Area Under the Curve [ROC] (AUC): {auc:.8f}")
    except ValueError as e:
        st.error(str(e))

# Calculate the EER
if st.button("🧮  Calculate the EER", key="eer_button"):
    try:
        equal_error_rate, equal_error_rate_threshold = eer(ss_lr_input, ds_lr_input)
        lg_equal_error_rate_threshold = np.log10(equal_error_rate_threshold)
        st.write(f"⚙️  Equal Error Rate (EER): {equal_error_rate:.8f}"
                 f"  \n⚙️  Equal Error Rate Threshold (EER-threshold): {equal_error_rate_threshold:.8f}"
                 f"  \n⚙️  Log10-EER-threshold: {lg_equal_error_rate_threshold:.8f}")
    except ValueError as e:
        st.error(str(e))

# Calculate the Cllr
if st.button("🧮  Calculate the Cllr", key="cllr_button"):
    st.latex(r'''
    C_{llr}=
    \frac{1}{2}\left[\frac{1}{N_{s}}\sum_{i=1}^{N_{s}}log_{2}\left(1+\frac{1}{LR_{si}}\right)
    +\frac{1}{N_{d}}\sum_{j=1}^{N_{d}}log_{2}\left ( 1+{LR_{dj}}\right)\right]
    ''')
    log_likelihood_ratio_cost = cllr(ss_lr_input, ds_lr_input)
    st.write(f"⚙️  Log-likelihood-ratio Cost (Cllr): {log_likelihood_ratio_cost:.8f}")

# Generate the ROC Curve
if st.button("📈  Generate the ROC Curve", key="roc_button"):
    try:
        roc_fig = plot_roc_curve(ss_lr_input, ds_lr_input, x_range_input_roc, y_range_input_roc, show_auc_input)
        st.pyplot(roc_fig)

        # Save the Figure into BytesIO for Downloading
        buf = io.BytesIO()
        roc_fig.savefig(buf, format='png', dpi=600)
        buf.seek(0)

        # Download Button
        st.download_button("💾  Download the ROC Curve", buf, "roc_curve.png", "image/png")
    except Exception as e:
        st.error(f"Error generating ROC curve: {str(e)}")

# Generate the DET Curve
if st.button("📈  Generate the DET Curve", key="det_button"):
    equal_error_rate = eer(ss_lr_input, ds_lr_input)[0]
    try:
        det_fig = plot_det_curve(ss_lr_input, ds_lr_input,
                                 equal_error_rate, x_range_input_det, y_range_input_det, show_eer_point_input)
        st.pyplot(det_fig)

        # Save the Figure into BytesIO for Downloading
        buf = io.BytesIO()
        det_fig.savefig(buf, format='png', dpi=600)
        buf.seek(0)

        # Download Button
        st.download_button("💾  Download the DET Curve", buf, "det_curve.png", "image/png")
    except Exception as e:
        st.error(f"Error generating DET curve: {str(e)}")

# Generate the Tippett Plot
if st.button("📈  Generate the Tippett Plot", key="tippett_button"):
    try:
        fig = tippett_plot(ss_lr_input, ds_lr_input, evi_value_input,
                           x_range_input_tippett, y_range_input_tippett,
                           ss_lr_tag_input, ds_lr_tag_input,
                           line_type_input,
                           legend_pos_input)
        st.pyplot(fig)

        # Save the Figure into BytesIO for Downloading
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=600)
        buf.seek(0)

        # Download Button
        st.download_button("💾  Download the Tippett Plot", buf, "tippett_plot.png", "image/png")
    except Exception as e:
        st.error(f"Error generating tippett plot: {str(e)}")

# streamlit run LR_Evaluator_v1.0.py
