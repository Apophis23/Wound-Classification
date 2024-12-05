import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 혼동 행렬 및 Classification Report 출력 함수
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def plot_classification_report(true_labels, pred_labels,class_names):
    return classification_report(true_labels, pred_labels, target_names = class_names)

def plot_distribution(data_frame, path='./image/class-distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data_frame, x='Class')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()