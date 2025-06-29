import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, average='macro', task_type='multiclass', num_classes=None):
    """
    统一评估函数，兼容多分类与多标签任务。
    
    Args:
        y_true (np.ndarray): shape=(N,) 或 (N, C)
        y_pred (np.ndarray): 同上
        average (str): 'macro'（默认）, 'micro', 'weighted'
        task_type (str): 'multiclass' 或 'multilabel'
        num_classes (int): 多分类任务中类别数（用于 AUROC）

    Returns:
        dict: 包含各类指标的结果字典
    """
    metrics = {}

    if task_type == 'multiclass':
        y_true = np.array(y_true)
        y_pred_label = np.argmax(y_pred, axis=1)

        metrics['accuracy'] = accuracy_score(y_true, y_pred_label)
        metrics['precision'] = precision_score(y_true, y_pred_label, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_label, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_label, average=average, zero_division=0)

        if num_classes is not None:
            try:
                metrics['auroc'] = roc_auc_score(
                    y_true=y_true,
                    y_score=y_pred,
                    multi_class='ovo',
                    average=average
                )
            except:
                metrics['auroc'] = -1.0  # 失败处理

    elif task_type == 'multilabel':
        y_pred_bin = (y_pred > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(y_true, y_pred_bin)
        metrics['precision'] = precision_score(y_true, y_pred_bin, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_bin, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_bin, average=average, zero_division=0)

        try:
            metrics['auroc'] = roc_auc_score(y_true, y_pred, average=average)
        except:
            metrics['auroc'] = -1.0

    return metrics
