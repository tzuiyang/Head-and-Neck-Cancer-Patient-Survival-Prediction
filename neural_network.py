import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = "data_sample(8000000 sample).csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Filter for Head and Neck Cancer based on Primary Site (ICD-O-3 codes: 000–149)
# Ensure Primary Site contains numeric codes
df['Primary Site'] = df['Primary Site'].astype(str).str.strip()  # Clean any spaces
df = df[df['Primary Site'].str.isdigit()]  # Keep rows with numeric Primary Site codes
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
# Replace placeholders ('Blank(s)', 'No/Unknown') with NaN and fill missing values with 0
X = X.replace(['Blank(s)', 'No/Unknown'], pd.NA).fillna(0)

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

# Step 6: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train the Neural Network
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', 
                          solver='adam', max_iter=500, random_state=42)

mlp_model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = mlp_model.predict(X_test)

# Step 10: Evaluate the Model
print("Neural Network Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

# Get prediction probabilities
y_pred_proba = mlp_model.predict_proba(X_test)[:, 1]

# Create a figure with subplots
plt.figure(figsize=(20, 10))

# 1. ROC Curve
plt.subplot(2, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# 2. Calibration Plot
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 3)
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Neural Network')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('True probability')
plt.title('Calibration Plot')
plt.legend()
