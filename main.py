%load_ext autoreload
%autoreload 2

!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "parkinsons.csv"  # Replace with the actual file path if different
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())
# Select input features (based on domain knowledge or EDA)
input_features = ['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']  # Example columns
output_feature = 'status'  # Assuming 'status' is the target column

X = df[input_features]
y = df[output_feature]
# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Split the dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Create an SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Ensure accuracy is at least 0.8
if accuracy >= 0.8:
    print("The model meets the accuracy requirement.")
else:
    print("The model does not meet the accuracy requirement.")
import joblib

# Save the model to a .joblib file
filename = 'parkinsons_model.joblib'
joblib.dump(model, filename)

# Upload the .joblib file to your GitHub repository (requires GitHub integration)
# Replace with your actual GitHub repository information
!git config --global user.email "your_email@example.com"
!git config --global user.name "Your Name"
!git add parkinsons_model.joblib
!git commit -m "Added trained model"
!git push origin main

# Update config.yaml
# You'll need to create a config.yaml file in the same directory
# Or replace '/content/config.yaml' with the correct path
with open('/content/config.yaml', 'w') as f:
  f.write(f"selected_features: {input_features}\n")
  f.write(f"path: {filename}\n")

print(f"Model saved as {filename} and uploaded to GitHub.")
print("config.yaml updated with selected features and model path.")
