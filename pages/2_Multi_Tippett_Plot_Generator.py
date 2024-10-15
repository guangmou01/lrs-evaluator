import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import io

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
    legend_pos_input = st.selectbox('Position of Legend', ['lower left', 'lower right',
                                                           'center left', 'center right',
                                                           'upper left', 'upper right'])

# Convert the Input String to a Floating Point Array
try:
    x_range_input = np.array([float(x) for x in x_range_input.split(',')])
    y_range_input = np.array([float(x) for x in y_range_input.split(',')])
except ValueError:
    st.error("Please enter valid numbers, separated by commas.")

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