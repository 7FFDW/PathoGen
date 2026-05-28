import os

import numpy as np
import random
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix

from sklearn.metrics import classification_report


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=20, verbose=False, count_loss=False):
        self.patience = patience
        self.warmup_epoch = warmup_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.model_path = model_path
        self.count_loss = count_loss

    def reset(self):
        self.counter = 0

    def __call__(self, epoch, val_loss, model, val_acc=None):
        flag = False
        if self.count_loss:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model)
                self.counter = 0
                flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, status='acc')
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch >= self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss'):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='macro')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat



    return acc, f1, roc_auc


def calculate_test_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes,
                           class_names=['Wild Type', 'L858R', 'Exon 19 Deletion', 'Other Mutations'], confusion_mat=False, roc_save_path=None,
                           state=None, fold=6):


    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    if num_classes == 2:
        class_names = ['wild', 'Mutations']
    targets_np = targets.cpu().numpy()
    _, pred_np = torch.max(logits, dim=1)
    pred_np = pred_np.cpu().numpy()
    probs_np = F.softmax(logits, dim=1).detach().cpu().numpy()


    report = classification_report(targets_np, pred_np, target_names=class_names, output_dict=True, zero_division=0)


    overall_acc = round(report['accuracy'], 4)
    macro_f1 = round(report['macro avg']['f1-score'], 4)


    per_class_f1 = {name: round(report[name]['f1-score'], 4) for name in class_names}


    mat = confusion_matrix(targets_np, pred_np)
    per_class_acc = {}
    for i in range(num_classes):
        tp = mat[i, i]
        tn = mat.sum() - (mat[i, :].sum() + mat[:, i].sum() - mat[i, i])
        accuracy = (tp + tn) / mat.sum() if mat.sum() > 0 else 0

        per_class_acc[class_names[i]] = round(accuracy, 4)


    mean_auc = 0.0
    per_class_auc = {name: 0.0 for name in class_names}

    if len(np.unique(targets_np)) == num_classes:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(targets_np, probs_np[:, 1])
            roc_val = auc(fpr, tpr)

            roc_val_rounded = round(roc_val, 4)
            mean_auc = roc_val_rounded
            per_class_auc[class_names[1]] = roc_val_rounded
            per_class_auc[class_names[0]] = roc_val_rounded

            if roc_save_path:
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {roc_val:.3f}', color='darkorange')  # 绘图时仍可使用原始精度以保证曲线平滑
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.tight_layout()
                os.makedirs(roc_save_path, exist_ok=True)
                plt.savefig(os.path.join(roc_save_path, f'{state}_roc_{fold}.jpg'), format='jpeg')
                plt.close()
        else:
            targets_bin = label_binarize(targets_np, classes=list(range(num_classes)))
            temp_roc_auc = {}

            # Create a figure for all class ROC curves
            plt.figure(figsize=(6, 5))

            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(targets_bin[:, i], probs_np[:, i])
                roc_val = auc(fpr, tpr)

                temp_roc_auc[i] = round(roc_val, 4)
                per_class_auc[class_names[i]] = temp_roc_auc[i]

                # Plot each class's ROC curve
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {temp_roc_auc[i]:.3f})')

            # Add diagonal line for random performance
            plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for All Classes')
            plt.legend(loc='lower right')
            plt.tight_layout()

            if roc_save_path:
                os.makedirs(roc_save_path, exist_ok=True)
                plt.savefig(os.path.join(roc_save_path, f'{state}_roc_all_{fold}.jpg'), dpi=300)
                plt.close()

            # 格式化平均AUC
            mean_auc = round(np.mean(list(temp_roc_auc.values())), 4)

    if confusion_mat:

        return overall_acc, macro_f1, mean_auc, per_class_acc,per_class_f1,per_class_auc,mat

    return overall_acc, macro_f1, mean_auc, per_class_acc,per_class_f1,per_class_auc


def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def draw_metrics(ts_writer, name, num_class, loss, acc, auc, mat, f1, fold):
    ts_writer.add_scalar("{}/loss".format(name), loss, fold)
    ts_writer.add_scalar("{}/acc".format(name), acc, fold)
    ts_writer.add_scalar("{}/auc".format(name), auc, fold)
    ts_writer.add_scalar("{}/f1".format(name), f1, fold)
    if mat is not None:
        ts_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class), fold)


def prepare_data(df, case_id, label_dict=None):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()

    slide_id = []
    label = []
    for case_id_ in case_id:
        idx = df_case_id.index(case_id_)
        slide_id.append(df_slide_id[idx])
        label_ = df_label[idx]
        if label_dict is None:
            label.append(int(label_))
        else:
            label.append(label_dict[label_])
    return slide_id, label


def return_splits(csv_path, label_dict=None, label_csv=None, test=False):
    split_df = pd.read_csv(csv_path)
    train_id = split_df['train'].dropna().tolist()
    val_id = split_df['val'].dropna().tolist()
    if test:
        test_id = split_df['test'].dropna().tolist()

    # 提取文本列
    train_text = split_df['train_text'].dropna().tolist()
    val_text = split_df['val_text'].dropna().tolist()
    if test:
        test_text = split_df['test_text'].dropna().tolist()

    if label_csv is None:
        train_label = split_df['train_label'].dropna().tolist()
        train_label = list(map(int, train_label))
        val_label = split_df['val_label'].dropna().tolist()
        val_label = list(map(int, val_label))
        if test:
            test_label = split_df['test_label'].dropna().tolist()
            test_label = list(map(int, test_label))
    else:
        df = pd.read_csv(label_csv)
        train_id, train_label = prepare_data(df, train_id, label_dict)
        val_id, val_label = prepare_data(df, val_id, label_dict)
        if test:
            test_id, test_label = prepare_data(df, test_id, label_dict)

    # 创建包含文本的元组
    train_data = list(zip(train_id, train_text, train_label))
    val_data = list(zip(val_id, val_text, val_label))

    if test:
        test_data = list(zip(test_id, test_text, test_label))
        return train_data, val_data, test_data

    return train_data, val_data





