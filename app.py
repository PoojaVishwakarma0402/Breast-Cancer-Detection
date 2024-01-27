import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the pre-trained XGBoost model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# Load a sample dataset or use your actual dataset loading logic
# Here, I'm using the breast cancer dataset for demonstration purposes
from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
X = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
y = pd.Series(cancer_dataset.target, name='target')

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Feature scaling (if needed)
# Note: You might need to apply the same preprocessing steps that were used during training
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_test_sc = sc.transform(X_test)

# Create a Streamlit web app
st.title("Breast Cancer Detection Web App")

# Display confusion matrix, accuracy, and classification report
st.subheader("Model Evaluation on Test Set:")

# Evaluate the model on the test set
y_pred = breast_cancer_detector_model.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
st.write("Accuracy:", accuracy)

# Display confusion matrix
st.write("Confusion Matrix:")
st.write(confusion_mat)

# Display classification report
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Additional information
st.subheader("Additional Information:")

# Display Cross-validation results
# Note: Ensure you have the necessary variables and models defined (e.g., xgb_classifier_pt)
# cross_validation = cross_val_score(estimator=xgb_classifier_pt, X=X_train_sc, y=y_train, cv=10)
# st.write("Mean Cross-validation accuracy:", cross_validation.mean())

# Save model button
if st.button("Save Model"):
    # Save the model with a new name
    new_model_name = "breast_cancer_detector_updated.pickle"
    pickle.dump(breast_cancer_detector_model, open(new_model_name, "wb"))
    st.success(f"Model saved as {new_model_name}")

# Show original model details
st.subheader("Original Model Details:")
st.write("Original model parameters:")
st.write(breast_cancer_detector_model.get_params())

# Show random search results
# st.subheader("Random Search Results:")
# st.write("Best parameters from Randomized Search:")
# st.write(random_search.best_params_)

# Show grid search results
# st.subheader("Grid Search Results:")
# st.write("Best parameters from Grid Search:")
# st.write(grid_search.best_params_)
