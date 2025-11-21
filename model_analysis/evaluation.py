# ==============================================================================
# EVALUATION - Đánh giá model chi tiết
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import streamlit as st

def generate_confusion_matrix(y_true, y_pred, labels, normalize=False):
    """
    Tạo confusion matrix đẹp
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        normalize: Normalize by row (True) or raw counts (False)
    
    Returns:
        Figure và confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    return fig, cm

def plot_roc_curves(y_true, y_pred_proba, labels, n_classes):
    """
    Plot ROC curves cho multi-class
    
    Args:
        y_true: True labels (integers)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        labels: Label names
        n_classes: Number of classes
    
    Returns:
        Figure
    """
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute ROC curve and ROC area for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multi-class', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig

def evaluation_report(y_true, y_pred, y_pred_proba, labels):
    """
    Tạo báo cáo đánh giá toàn diện
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        labels: Label names
    
    Returns:
        Dictionary với metrics
    """
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    
    return {
        'overall_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc
    }

def find_misclassified_samples(y_true, y_pred, X_data, top_n=10):
    """
    Tìm các samples bị misclassified
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X_data: Input data
        top_n: Top N worst predictions
    
    Returns:
        Indices và info của misclassified samples
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        return []
    
    # Get confidence of predictions
    results = []
    for idx in misclassified_idx[:top_n]:
        results.append({
            'index': idx,
            'true_label': y_true[idx],
            'pred_label': y_pred[idx],
            'image': X_data[idx] if X_data is not None else None
        })
    
    return results

def plot_per_class_metrics(metrics_dict, labels):
    """
    Plot per-class precision, recall, f1-score
    
    Args:
        metrics_dict: Output từ evaluation_report()
        labels: Label names
    
    Returns:
        Figure
    """
    report = metrics_dict['classification_report']
    
    # Extract metrics
    precisions = [report[str(label)]['precision'] for label in labels]
    recalls = [report[str(label)]['recall'] for label in labels]
    f1_scores = [report[str(label)]['f1-score'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig

def display_evaluation_streamlit(metrics, labels, y_pred_proba=None):
    """
    Hiển thị evaluation dashboard trong Streamlit
    
    Args:
        metrics: Output từ evaluation_report()
        labels: Label names
        y_pred_proba: Predicted probabilities (optional, for ROC)
    """
    st.markdown("### 📊 MODEL EVALUATION DASHBOARD")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", f"{metrics['overall_accuracy']*100:.2f}%")
    
    report = metrics['classification_report']
    with col2:
        st.metric("Macro Avg Precision", f"{report['macro avg']['precision']*100:.2f}%")
    
    with col3:
        st.metric("Macro Avg Recall", f"{report['macro avg']['recall']*100:.2f}%")
    
    st.markdown("---")
    
    # Per-class accuracy
    st.markdown("#### 🎯 Per-Class Accuracy")
    per_class_df = {
        'Class': labels,
        'Accuracy': [f"{acc*100:.2f}%" for acc in metrics['per_class_accuracy']],
        'Precision': [f"{report[str(label)]['precision']*100:.2f}%" for label in labels],
        'Recall': [f"{report[str(label)]['recall']*100:.2f}%" for label in labels],
        'F1-Score': [f"{report[str(label)]['f1-score']*100:.2f}%" for label in labels]
    }
    
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)
    
    # Confusion Matrix
    st.markdown("#### 🔲 Confusion Matrix")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cm, _ = generate_confusion_matrix(
            np.arange(len(labels)),  # Dummy for display
            np.arange(len(labels)),
            labels,
            normalize=False
        )
        st.pyplot(fig_cm)
        st.caption("Raw counts")
    
    with col2:
        fig_cm_norm, _ = generate_confusion_matrix(
            np.arange(len(labels)),
            np.arange(len(labels)),
            labels,
            normalize=True
        )
        st.pyplot(fig_cm_norm)
        st.caption("Normalized by row")
    
    # ROC Curves (if probabilities provided)
    if y_pred_proba is not None:
        st.markdown("#### 📈 ROC Curves")
        # Would need actual data to plot
        st.info("ROC curves require full test set predictions")

__all__ = [
    'generate_confusion_matrix',
    'plot_roc_curves',
    'evaluation_report',
    'find_misclassified_samples',
    'plot_per_class_metrics',
    'display_evaluation_streamlit'
]
