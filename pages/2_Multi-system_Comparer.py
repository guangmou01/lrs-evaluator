import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io

def auc(ss_lr, ds_lr):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    auc_value = roc_auc_score(labels, scores)

    return auc_value

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

def calculate_metrics(system_name, ss_lr, ds_lr):
    try:
        ss_lr = np.array([float(x) for x in ss_lr.split(',')])
        ds_lr = np.array([float(x) for x in ds_lr.split(',')])
        eer_value, eer_threshold = eer(ss_lr, ds_lr)
        cllr_value = cllr(ss_lr, ds_lr)
        auc_value = auc(ss_lr, ds_lr)
        return {
            "System": system_name,
            "EER": eer_value,
            "EER-threshold": eer_threshold,
            "Log10-EER-threshold": np.log10(eer_threshold),
            "Cllr": cllr_value,
            "AUC": auc_value
        }
    except Exception as e:
        st.error(f"Error calculating metrics for {system_name}: {str(e)}")
        return None

def muti_tippett_plot(n_1, ss_lr_1, ds_lr_1, line_type_1,
                      n_2, ss_lr_2, ds_lr_2, line_type_2,
                      n_3, ss_lr_3, ds_lr_3, line_type_3,
                      n_4, ss_lr_4, ds_lr_4, line_type_4,
                      n_5, ss_lr_5, ds_lr_5, line_type_5,
                      evidence_lr,
                      x_range, y_range,
                      ss_lr_tag, ds_lr_tag,
                      legend_pos):

    fig_muti_tippett, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Log10 Likelihood Ratio')
    ax.set_ylabel('Cumulative Proportion')
    ax.grid(True, alpha=0.4)

    if n_1 != "NO":
        try:
            ss_lr_1 = np.array([float(x) for x in ss_lr_1.split(',')])
            ds_lr_1 = np.array([float(x) for x in ds_lr_1.split(',')])
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")
        ss_lr_1_sorted = np.sort(np.log10(ss_lr_1))
        ss_1_cumulative = np.arange(1, len(ss_lr_1_sorted) + 1) / len(ss_lr_1_sorted)
        ds_lr_1_sorted = np.sort(np.log10(ds_lr_1))[::-1]
        ds_1_cumulative = np.arange(1, len(ds_lr_1_sorted) + 1) / len(ds_lr_1_sorted)
        ax.plot(ds_lr_1_sorted, ds_1_cumulative, color='blue', linestyle=line_type_1)
        ax.plot(ss_lr_1_sorted, ss_1_cumulative, color='red', linestyle=line_type_1)

    if n_2 != "No":
        try:
            ss_lr_2 = np.array([float(x) for x in ss_lr_2.split(',')])
            ds_lr_2 = np.array([float(x) for x in ds_lr_2.split(',')])
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")
        ss_lr_2_sorted = np.sort(np.log10(ss_lr_2))
        ss_2_cumulative = np.arange(1, len(ss_lr_2_sorted) + 1) / len(ss_lr_2_sorted)
        ds_lr_2_sorted = np.sort(np.log10(ds_lr_2))[::-1]
        ds_2_cumulative = np.arange(1, len(ds_lr_2_sorted) + 1) / len(ds_lr_2_sorted)
        ax.plot(ds_lr_2_sorted, ds_2_cumulative, color='blue', linestyle=line_type_2)
        ax.plot(ss_lr_2_sorted, ss_2_cumulative, color='red', linestyle=line_type_2)

    if n_3 != "No":
        try:
            ss_lr_3 = np.array([float(x) for x in ss_lr_3.split(',')])
            ds_lr_3 = np.array([float(x) for x in ds_lr_3.split(',')])
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")
        ss_lr_3_sorted = np.sort(np.log10(ss_lr_3))
        ss_3_cumulative = np.arange(1, len(ss_lr_3_sorted) + 1) / len(ss_lr_3_sorted)
        ds_lr_3_sorted = np.sort(np.log10(ds_lr_3))[::-1]
        ds_3_cumulative = np.arange(1, len(ds_lr_3_sorted) + 1) / len(ds_lr_3_sorted)
        ax.plot(ds_lr_3_sorted, ds_3_cumulative, color='blue', linestyle=line_type_3)
        ax.plot(ss_lr_3_sorted, ss_3_cumulative, color='red', linestyle=line_type_3)

    if n_4 != "No":
        try:
            ss_lr_4 = np.array([float(x) for x in ss_lr_4.split(',')])
            ds_lr_4 = np.array([float(x) for x in ds_lr_4.split(',')])
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")
        ss_lr_4_sorted = np.sort(np.log10(ss_lr_4))
        ss_4_cumulative = np.arange(1, len(ss_lr_4_sorted) + 1) / len(ss_lr_4_sorted)
        ds_lr_4_sorted = np.sort(np.log10(ds_lr_4))[::-1]
        ds_4_cumulative = np.arange(1, len(ds_lr_4_sorted) + 1) / len(ds_lr_4_sorted)
        ax.plot(ds_lr_4_sorted, ds_4_cumulative, color='blue', linestyle=line_type_4)
        ax.plot(ss_lr_4_sorted, ss_4_cumulative, color='red', linestyle=line_type_4)

    if n_5 != "No":
        try:
            ss_lr_5 = np.array([float(x) for x in ss_lr_5.split(',')])
            ds_lr_5 = np.array([float(x) for x in ds_lr_5.split(',')])
        except ValueError:
            st.error("Please enter valid numbers, separated by commas.")
        ss_lr_5_sorted = np.sort(np.log10(ss_lr_5))
        ss_5_cumulative = np.arange(1, len(ss_lr_5_sorted) + 1) / len(ss_lr_5_sorted)
        ds_lr_5_sorted = np.sort(np.log10(ds_lr_5))[::-1]
        ds_5_cumulative = np.arange(1, len(ds_lr_5_sorted) + 1) / len(ds_lr_5_sorted)
        ax.plot(ds_lr_5_sorted, ds_5_cumulative, color='blue', linestyle=line_type_5)
        ax.plot(ss_lr_5_sorted, ss_5_cumulative, color='red', linestyle=line_type_5)

    if evidence_lr != "None":
        evidence_lr = float(evidence_lr)
        ax.axvline(np.log10(evidence_lr), color='green', linestyle='-', alpha=0.6)  # Draw the Evidence Line
        ax.annotate(f'E = {evidence_lr}',
                    xy=(np.log10(evidence_lr), 0.5),
                    xytext=(np.log10(evidence_lr)+(max(x_range)-min(x_range))/50, 0.5),
                    fontsize=10, ha='left', color='black')
        # Annotate the Evidence Value

    handles = [Line2D([0], [0], color='blue', lw=1, linestyle='-', label=ds_lr_tag),
               Line2D([0], [0], color='red', lw=1, linestyle='-', label=ss_lr_tag)]
    if legend_pos != 'None':
        ax.legend(handles=handles, loc=legend_pos)

    return fig_muti_tippett

# Header
st.set_page_config(page_title="LRs Evaluator",
                   page_icon="⚖️")
st.title("⚖️ Multi Tippett Plot Generator")
st.write("Author: Guangmou"
         "  \n E-mail: forensicstats@hotmail.com")
st.markdown("---")

# Get Input LRs and Other Parameters
with st.expander('📊  System-1 Setting'):
    n_1_input = st.selectbox('Whether To Evaluate a System 1:', ['Yes', 'No'], index=0)
    ss_lr_1_input = st.text_input('LR Values of Positive Pairs from System-1',
                                  '0.8, 1, 5, 3, 9, 10, 25, 6, 20, 18, 0.9, 0.7, 11')
    ds_lr_1_input = st.text_input('LR Values of Negative Pairs from System-1',
                                  '0.002, 0.01, 0.3, 0.5, 0.9, 1.2, 0.6, 0.05, 0.006, 1.3, 0.4, 0.2, 0.03, 1.1, 2, 0.0005')
    line_type_1_input = st.selectbox('Line Type for System-1', ['solid', 'dotted', 'dashed', 'dash-dot'])
    # Convert the Line Type
    if line_type_1_input == 'solid':
        line_type_1_input = '-'
    elif line_type_1_input == 'dotted':
        line_type_1_input = ':'
    elif line_type_1_input == 'dashed':
        line_type_1_input = '--'
    elif line_type_1_input == 'dash-dot':
        line_type_1_input = '-.'

with st.expander('📊  System-2 Setting'):
    n_2_input = st.selectbox('Whether To Evaluate a System 2:', ['Yes', 'No'], index=1)
    ss_lr_2_input = st.text_input('LR Values of Positive Pairs from System-2')
    ds_lr_2_input = st.text_input('LR Values of Negative Pairs from System-2')
    line_type_2_input = st.selectbox('Line Type for System-2', ['None', 'solid', 'dotted', 'dashed', 'dash-dot'])
    # Convert the Line Type
    if line_type_2_input == 'solid':
        line_type_2_input = '-'
    elif line_type_2_input == 'dotted':
        line_type_2_input = ':'
    elif line_type_2_input == 'dashed':
        line_type_2_input = '--'
    elif line_type_2_input == 'dash-dot':
        line_type_2_input = '-.'

with st.expander('📊  System-3 Setting'):
    n_3_input = st.selectbox('Whether To Evaluate a System 3:', ['Yes', 'No'], index=1)
    ss_lr_3_input = st.text_input('LR Values of Positive Pairs from System-3')
    ds_lr_3_input = st.text_input('LR Values of Negative Pairs from System-3')
    line_type_3_input = st.selectbox('Line Type for System-3', ['None', 'solid', 'dotted', 'dashed', 'dash-dot'])
    # Convert the Line Type
    if line_type_3_input == 'solid':
        line_type_3_input = '-'
    elif line_type_3_input == 'dotted':
        line_type_3_input = ':'
    elif line_type_3_input == 'dashed':
        line_type_3_input = '--'
    elif line_type_3_input == 'dash-dot':
        line_type_3_input = '-.'

with st.expander('📊  System-4 Setting'):
    n_4_input = st.selectbox('Whether To Evaluate a System 4:', ['Yes', 'No'], index=1)
    ss_lr_4_input = st.text_input('LR Values of Positive Pairs from System-4')
    ds_lr_4_input = st.text_input('LR Values of Negative Pairs from System-4')
    line_type_4_input = st.selectbox('Line Type for System-4', ['None', 'solid', 'dotted', 'dashed', 'dash-dot'])
    # Convert the Line Type
    if line_type_4_input == 'solid':
        line_type_4_input = '-'
    elif line_type_4_input == 'dotted':
        line_type_4_input = ':'
    elif line_type_4_input == 'dashed':
        line_type_4_input = '--'
    elif line_type_4_input == 'dash-dot':
        line_type_4_input = '-.'

with st.expander('📊  System-5 Setting'):
    n_5_input = st.selectbox('Whether To Evaluate a System 5:', ['Yes', 'No'], index=1)
    ss_lr_5_input = st.text_input('LR Values of Positive Pairs from System-5')
    ds_lr_5_input = st.text_input('LR Values of Negative Pairs from System-5')
    line_type_5_input = st.selectbox('Line Type for System-5', ['None', 'solid', 'dotted', 'dashed', 'dash-dot'])
    # Convert the Line Type
    if line_type_5_input == 'solid':
        line_type_5_input = '-'
    elif line_type_5_input == 'dotted':
        line_type_5_input = ':'
    elif line_type_5_input == 'dashed':
        line_type_5_input = '--'
    elif line_type_5_input == 'dash-dot':
        line_type_5_input = '-.'

with st.expander('⚙️  Tippett Setting'):
    evi_value_input = st.text_input('Input the Evidence LR ("None" or input a valid positive number)',
                                    'None')
    col1, col2 = st.columns(2)
    with col1:
        x_range_input = st.text_input('X Range (e.g. -3.5, 1.5)', '-3.5, 1.5', key="x_range_tippett")
    with col2:
        y_range_input = st.text_input('Y Range (e.g. 0, 1)', '0, 1', key="y_range_tippett")
    col1, col2 = st.columns(2)
    with col1:
        ss_lr_tag_input = st.text_input('Name the Tag for Positive Pairs', 'Same Source')
    with col2:
        ds_lr_tag_input = st.text_input('Name the Tag for Negative Pairs', 'Different Source')
    legend_pos_input = st.selectbox('Position of Legend', ['None',
                                                           'lower left', 'lower right',
                                                           'center left', 'center right',
                                                           'upper left', 'upper right'])

# Convert the Input String to a Floating Point Array
try:
    x_range_input = np.array([float(x) for x in x_range_input.split(',')])
    y_range_input = np.array([float(x) for x in y_range_input.split(',')])
except ValueError:
    st.error("Please enter valid numbers, separated by commas.")

# Generate Numeric Metrics
if st.button("🧮  Generate Numeric Metrics"):
    metrics = []

    # System 1
    if n_1_input == "Yes":
        system_1_metrics = calculate_metrics("System 1", ss_lr_1_input, ds_lr_1_input)
        if system_1_metrics:
            metrics.append(system_1_metrics)

    # System 2
    if n_2_input == "Yes":
        system_2_metrics = calculate_metrics("System 2", ss_lr_2_input, ds_lr_2_input)
        if system_2_metrics:
            metrics.append(system_2_metrics)

    # System 3
    if n_3_input == "Yes":
        system_3_metrics = calculate_metrics("System 3", ss_lr_3_input, ds_lr_3_input)
        if system_3_metrics:
            metrics.append(system_3_metrics)

    # System 4
    if n_4_input == "Yes":
        system_4_metrics = calculate_metrics("System 4", ss_lr_4_input, ds_lr_4_input)
        if system_4_metrics:
            metrics.append(system_4_metrics)

    # System 5
    if n_5_input == "Yes":
        system_5_metrics = calculate_metrics("System 5", ss_lr_5_input, ds_lr_5_input)
        if system_5_metrics:
            metrics.append(system_5_metrics)

    # Create and Display DataFrame
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)

        # Allow the user to download the metrics as CSV
        csv = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Metrics as CSV", data=csv, file_name="metrics.csv", mime="text/csv")
    else:
        st.warning("No metrics calculated. Ensure at least one system is selected and has valid inputs.")


# Generate the Tippett Plot
if st.button("📈  Generate the Multi Tippett Plot"):
    try:
        tippett_fig = muti_tippett_plot(n_1_input, ss_lr_1_input, ds_lr_1_input, line_type_1_input,
                                        n_2_input, ss_lr_2_input, ds_lr_2_input, line_type_2_input,
                                        n_3_input, ss_lr_3_input, ds_lr_3_input, line_type_3_input,
                                        n_4_input, ss_lr_4_input, ds_lr_4_input, line_type_4_input,
                                        n_5_input, ss_lr_5_input, ds_lr_5_input, line_type_5_input,
                                        evi_value_input,
                                        x_range_input, y_range_input,
                                        ss_lr_tag_input, ds_lr_tag_input,
                                        legend_pos_input)
        st.pyplot(tippett_fig)

        # Save the Figure into BytesIO for Downloading
        buf = io.BytesIO()
        tippett_fig.savefig(buf, format='png', dpi=600)
        buf.seek(0)

        # Download Button
        st.download_button("💾  Download the Multi Tippett Plot", buf, "multi_tippett_plot.png", "image/png")
    except Exception as e:
        st.error(f"Error generating tippett plot: {str(e)}")
