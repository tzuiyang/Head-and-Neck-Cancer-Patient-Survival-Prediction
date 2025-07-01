import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
file_path = "data_sample(8000000 sample).csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Step 2: Filter for Head and Neck Cancer based on Primary Site (ICD-O-3 codes: 000–149)
head_neck_codes = list(range(0, 150))  # Primary Site codes from 000 to 149
df = df[df['Primary Site'].apply(lambda x: str(x).isdigit() and int(x) in head_neck_codes)]

# Step 3: Encode the target variable ('Alive' -> 0, 'Dead' -> 1)
df['Vital status recode (study cutoff used)'] = df['Vital status recode (study cutoff used)'].apply(
    lambda x: 1 if x.strip() == 'Dead' else 0
)

# Step 4: Select features (X) and target variable (y)
X = df[['Age', 'Sex', 'Year of diagnosis', 'Race', 'Primary Site',
        'Combined Summary Stage (2004+)', 'Surgery of oth reg/dis sites (1998-2002)',
        'Chemotherapy', 'Radiation recode', 'CS tumor size (2004-2015)',
        'CS lymph nodes (2004-2015)']]

y = df['Vital status recode (study cutoff used)']

# Step 5: Handle missing or invalid values
X = X.replace(to_replace=['Blank(s)', 'No/Unknown'], value=pd.NA)
X = X.fillna({
    'CS tumor size (2004-2015)': 0,
    'CS lymph nodes (2004-2015)': 0,
    'Combined Summary Stage (2004+)': 'Unknown',
    'Surgery of oth reg/dis sites (1998-2002)': 'Unknown',
    'Chemotherapy': 'Unknown',
    'Radiation recode': 'Unknown'
})

# Step 6: Convert categorical features to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
print("Filtered Data Size for Head and Neck Cancer:", df.shape[0])
print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# plots to illsutrate how model performed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
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

# Calibration Plot
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Plot')
plt.show()
