# LR-based System Evaluator (LRs-Evaluator)

## Abstract
This script is based on Streamlit to provide a convenient tool for relevant practitioners to evaluate the performance (mainly validity) of LR-based systems in forensic practice and forensic research activities.

## Basic Functions

- **Area Under the Curve (AUC) Calculation**: 
  Computes the AUC for the ROC curve to assess the model's classification performance, with values ranging from 0 to 1.

- **Equal Error Rate (EER) Calculation**: 
  Computes the EER, which is the point at which the false positive rate (FPR) equals the false negative rate (FNR), indicating the balance point of the model.

- **Log-likelihood-ratio Cost (Cllr) Calculation**: 
  Computes the cost of using the log-likelihood ratio for classification, providing an indication of the model's precision.

- **Generate the ROC Curve**: 
  Visualizes the trade-off between true positive rate (TPR) and false positive rate (FPR) at various threshold settings.

- **Generate the DET Curve**: 
  Displays the relationship between false positive rate (FPR) and false negative rate (FNR), providing insight into the detection error characteristics of the system.

- **Generate the Tippett Plot**: 
  Illustrates the cumulative distribution of likelihood ratios for both positive and negative pairs, aiding in evidence evaluation.

## Numeric-metrics Approach:
**AUC**:
```python
def auc(ss_lr, ds_lr):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    auc_value = roc_auc_score(labels, scores)

    return auc_value
```

**EER**:
```python
def eer(ss_lr, ds_lr):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_value = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    return eer_value, eer_threshold
```

**Cllr**:
```python
def cllr(ss_lr, ds_lr):
    punish_ss = np.log2(1 + (1 / ss_lr))
    punish_ds = np.log2(1 + ds_lr)
    n_vali_ss = len(ss_lr)
    n_vali_ds = len(ds_lr)
    cllr_value = 0.5 * (1 / n_vali_ss * sum(punish_ss) + 1 / n_vali_ds * sum(punish_ds))

    return cllr_value
```

## Graphic-metrics Approach:
**ROC Curve**:
```python
def plot_roc_curve(ss_lr, ds_lr, x_range, y_range, show_auc):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    auc_value = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    fig_roc, ax = plt.subplots(figsize=(8, 8))
    # Draw the ROC Curve
    ax.plot(fpr, tpr, label='ROC Curve', color='red')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.legend(loc='lower right')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.4)
    if show_auc == "Yes":
        ax.annotate(f'AUC = {auc_value:.4f}',
                    xy=(0.61, 0.45),
                    fontsize=10,
                    color='black')
        # Annotate the AUC Value

    return fig_roc
```

**DET Curve**:
```python
def plot_det_curve(ss_lr, ds_lr, eer_value, x_range, y_range, show_eer_point):
    scores = np.concatenate([ss_lr, ds_lr])
    labels = np.concatenate([np.ones_like(ss_lr), np.zeros_like(ds_lr)])
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fig_det, ax = plt.subplots(figsize=(8, 8))
    # Draw the DET Curve
    ax.plot(fpr, fnr, label='DET Curve', color='blue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('False Negative Rate (FNR)')
    ax.legend(loc='upper right')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, alpha=0.4)
    if show_eer_point == "Yes":
        ax.scatter(eer_value, eer_value, s=20, color='red', label="EER Point", marker='o', zorder=8)
        # Show the ERR Point

    return fig_det
```

**Tippett Plot**:
```python
def tippett_plot(ss_lr, ds_lr, evidence_lr,
                 x_range, y_range,
                 ss_lr_tag, ds_lr_tag,
                 line_type,
                 legend_pos):
    ss_lr_sorted = np.sort(np.log10(ss_lr))
    ss_cumulative = np.arange(1, len(ss_lr_sorted) + 1) / len(ss_lr_sorted)
    ds_lr_sorted = np.sort(np.log10(ds_lr))[::-1]
    ds_cumulative = np.arange(1, len(ds_lr_sorted) + 1) / len(ds_lr_sorted)

    fig_tippett, ax = plt.subplots(figsize=(8, 6))
    # Draw the Tippett Plot
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
        ax.axvline(np.log10(evidence_lr), color='green', linestyle='-', alpha=0.6)  # Draw the Evidence Line
        ax.annotate(f'E = {evidence_lr}',
                    xy=(np.log10(evidence_lr), 0.5),
                    xytext=(np.log10(evidence_lr)+(max(x_range)-min(x_range))/50, 0.5),
                    fontsize=10, ha='left', color='black')
        # Annotate the Evidence Value

    return fig_tippett
```
