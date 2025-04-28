Wasel Hand Gesture Recognition – Model Files
**Description**
This repository contains the final trained machine learning model (wasel_rf.pkl) and the feature scaler (scaler2.pkl) for my senior project: Wasel – Real-Time Hand Gesture Recognition Using Random Forest Classifier.
The model was developed using custom hand landmarks data extracted through computer vision techniques to classify gestures accurately in real-time.

**Files**
wasel_rf.pkl: Trained Random Forest model used for gesture prediction.
scaler2.pkl: StandardScaler object used to normalize input features before feeding them into the model.

**How to use**
import pickle

# Load scaler
with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load model
with open('wasel_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Example usage
X = scaler.transform([[feature1, feature2, feature3, ..., featureN]])
prediction = model.predict(X)

**Notes**
This repository is intended for academic demonstration purposes only.
Full dataset and training code are not included here.
**Acknowledgment**
Developed as part of the Senior Project requirement at King Abdulaziz University.
## Contact
for inquiries or collaboration opportunities, feel free to reach out:
Email: sarahalkabkabi@gmail.com
