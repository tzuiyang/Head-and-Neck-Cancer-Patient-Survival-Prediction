import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
file_path = "data_sample(8000000 sample).csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Filter for Head and Neck Cancer based on Primary Site (ICD-O-3 codes: 000–149)
# Ensure Primary Site contains numeric codes
df['Primary Site'] = df['Primary Site'].astype(str).str.strip()  # Remove any leading/trailing spaces
df = df[df['Primary Site'].str.isdigit()]  # Keep only numeric rows
df['Primary Site'] = df['Primary Site'].astype(int)  # Convert to integers

# Filter for relevant Head and Neck Cancer codes
head_neck_codes = list(range(0, 150))  # Codes from 000 to 149
df = df[df['Primary Site'].isin(head_neck_codes)]
print(f"Filtered dataset size for Head and Neck Cancer: {df.shape[0]} rows")

# Step 3: Encode the target variable ('Alive' -> 0, 'Dead' -> 1)
df['Vital status recode (study cutoff used)'] = df['Vital status recode (study cutoff used)'].apply(
    lambda x: 1 if x.strip() == 'Dead' else 0
)

# Step 4: Prepare features (X) and target (y)
X = df[['Age', 'Sex', 'Year of diagnosis', 'Race', 'Primary Site',
        'Combined Summary Stage (2004+)', 'Surgery of oth reg/dis sites (1998-2002)',
        'Chemotherapy', 'Radiation recode', 'CS tumor size (2004-2015)',
        'CS lymph nodes (2004-2015)']]

y = df['Vital status recode (study cutoff used)']

# Step 5: Clean and preprocess the data
# Replace 'Blank(s)' or 'No/Unknown' with NaN and fill missing values with 0 or "Unknown"
X = X.replace(['Blank(s)', 'No/Unknown'], pd.NA).fillna(0)

# Convert categorical columns into dummy variables
X = pd.get_dummies(X, drop_first=True)

# Step 6: Standardize numerical features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train the Support Vector Classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)  # RBF kernel for non-linear data
svm_model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = svm_model.predict(X_test)

# Step 10: Evaluate the model
print("SVM Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# plots
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 1. ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, svm_model.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 2. Model Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, svm_model.predict_proba(X_test)[:, 1], n_bins=10)
plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Plot')
plt.show()
