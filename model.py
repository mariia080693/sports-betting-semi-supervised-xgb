import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'lines.markersize': 6,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'figure.autolayout': True
})

# Load and prepare data
df = pd.read_csv("data.csv")
features = [
    "Age", "State", "RegistrationDevice", "FirstBetDevice",
    "AcquisitionSource", "MainBetSport", "FirstWeekTurnover",
    "DaysReg", "DaysToFirstBet"
]

X = df[features].copy()
# Convert object columns to category dtype for XGBoost native handling
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# Split labeled/unlabeled
has_label = df["SurveyAnswer"].notna()
X_labeled, y_labeled = X.loc[has_label], df.loc[has_label, "SurveyAnswer"].values
X_unlabeled = X.loc[~has_label]

# Encode target labels
le = LabelEncoder()
y_labeled_enc = le.fit_transform(y_labeled)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_labeled, y_labeled_enc, test_size=0.3, random_state=42, stratify=y_labeled_enc
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Unlabeled: {len(X_unlabeled)}\n")
#print(f"First training sample X: \n{X_train.iloc[0]},\n Y: {y_train[0]}\n")

# Compute sample weights for class imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
sample_weights = np.array([class_weights[label] for label in y_train])

# Helper functions
def select_confident_indices(proba, thresholds):
    preds = np.argmax(proba, axis=1)   # predicted class for each sample
    confident_idx = []

    for i, p in enumerate(proba):
        cls = preds[i]
        th = thresholds.get(cls, 1.0)  # default to 1.0 if no threshold for this class
        if p[cls] >= th:
            confident_idx.append(True)
        else:
            confident_idx.append(False)

    return np.array(confident_idx), preds

def plot_per_class_f1(history, labels):
    history = np.array(history)
    plt.figure(figsize=(8,5))
    for idx, label in enumerate(labels):
        plt.plot(history[:, idx], label=f"{label} F1")
    plt.xlabel("Iteration")
    plt.ylabel("F1-score")
    plt.title("Per-class F1 During Self-Training")
    plt.legend()
    plt.grid(True)
    plt.show()

# Semi-supervised XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    early_stopping_rounds=20,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=True,
    reg_lambda=1,
    random_state=42
)

val_macro_f1_history, val_per_class_f1_history = [], []
class_thresholds = {0: 0.8, 1: 0.6, 2: 0.6}  # Confidence thresholds for pseudo-labeling 

for i in range(20):
    # Compute sample weights for current training set (including pseudo-labels in later iterations)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[label] for label in y_train])

    xgb_clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Validation metrics
    val_pred = xgb_clf.predict(X_val)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    per_class_f1 = f1_score(y_val, val_pred, average=None, labels=np.unique(y_val))
    val_macro_f1_history.append(val_macro_f1)
    val_per_class_f1_history.append(per_class_f1)

    per_class_dict = {le.inverse_transform([cls])[0]: round(float(score), 3)
                  for cls, score in zip(np.unique(y_val), per_class_f1)}
    print(f"Iteration {i+1} | Macro F1: {val_macro_f1:.3f} | Per-class F1: {per_class_dict}")

    if len(X_unlabeled) == 0:
        break

    # Pseudo-label high-confidence samples
    proba = xgb_clf.predict_proba(X_unlabeled)
    confident_idx, preds = select_confident_indices(proba, class_thresholds)
    if not confident_idx.any():
       break

    X_train = pd.concat([X_train, X_unlabeled[confident_idx]], ignore_index=True)
    y_train = np.concatenate([y_train, preds[confident_idx]])
    X_unlabeled = X_unlabeled[~confident_idx]

    print(f" → Added {confident_idx.sum()} labels, {len(X_unlabeled)} unlabeled remaining")

# Plot per-class F1
plot_per_class_f1(val_per_class_f1_history, le.inverse_transform(np.unique(y_val)))

# Final evaluation
test_pred = xgb_clf.predict(X_test)
print("\nFinal Test Performance:")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=np.unique(y_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Greens", values_format="d")
plt.title("Confusion Matrix - Final Test Set")
plt.show()

# Save the final trained model
import pickle
xgb_clf.save_model("trained_model.json")
print("Model saved as 'trained_model.json'")

# Save the label encoder for future predictions
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved as 'label_encoder.pkl'")
