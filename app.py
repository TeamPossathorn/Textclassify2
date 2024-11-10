import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(correct_tags, predicted_tags):
    labels = sorted(set(correct_tags) | set(predicted_tags))

    # ตรวจสอบว่า correct_tags และ predicted_tags มีข้อมูลก่อนที่จะสร้าง Confusion Matrix
    if len(correct_tags) == 0 or len(predicted_tags) == 0:
        st.error("Error: No correct or predicted tags available for confusion matrix.")
        return

    # สร้าง Confusion Matrix
    cm = confusion_matrix(correct_tags, predicted_tags, labels=labels)
    
    fig, ax = plt.subplots()
    
    # สร้าง mask สำหรับค่าในเส้นทแยงมุม (ค่าที่ทำนายถูก) และค่าที่ไม่อยู่ในเส้นทแยงมุม (ค่าที่ทำนายผิด)
    mask_correct = np.eye(len(cm), dtype=bool)  # ค่าในเส้นทแยงมุม
    mask_incorrect = ~mask_correct  # ค่าอื่นๆ ที่ไม่ใช่เส้นทแยงมุม

    # Plot heatmap สำหรับค่าที่ทำนายถูก (แสดงด้วยสีน้ำเงิน)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="Blues", mask=mask_incorrect, cbar=False, ax=ax)

    # Plot heatmap สำหรับค่าที่ทำนายผิด (แสดงด้วยสีแดง)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="Reds", mask=mask_correct, cbar=False, ax=ax)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
