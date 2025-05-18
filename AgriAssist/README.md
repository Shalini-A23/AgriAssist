# Crop Recommendation System Using Machine Learning
# Description
The Crop Recommendation System is a machine learning-based application that provides recommendations for suitable crops based on various environmental and soil conditions. It aims to assist farmers and agricultural professionals in making informed decisions about crop selection, optimizing yields, and maximizing profitability.

The system takes into account several factors such as soil type, climate, rainfall, temperature, humidity, and pH levels to determine the most suitable crops for a given region. By analyzing historical data and using predictive models, the system provides personalized recommendations tailored to the specific conditions of a farm or agricultural area.

# Theoretical Overview
This system recommends the most suitable crop to grow based on soil and environmental parameters using machine learning. It improves accuracy by augmenting the dataset with synthetic examples for certain crops and uses the XGBoost classifier for high performance.

1. Dataset Preparation
The base dataset is loaded from Crop_recommendation.csv, which includes:
Features: N, P, K, temperature, humidity, ph, and rainfall
Label: Crop name (label column)
2. Synthetic Data Augmentation
Some crops are underrepresented, so synthetic data is generated for five crops: rice, maize, jute, blackgram, and coconut.
Each crop is assigned realistic value ranges for all features based on agronomic knowledge.
50 new data points are generated per crop using:
np.random.randint() for nutrients
np.random.uniform() for continuous values like temperature and rainfall
These synthetic samples are then appended to the original dataset to improve class balance.
3. Label Encoding
The label (crop name) is encoded into numerical values using LabelEncoder.
This is necessary because ML models work with numeric data.
4. Feature Scaling
Standardization is applied using StandardScaler to ensure all input features contribute equally during training.
This process transforms the features to have zero mean and unit variance.
5. Train-Test Split
The dataset is split into training (80%) and testing (20%) sets.
stratify=y ensures that the class distribution is maintained across both sets.
6. Model Training (XGBoost Classifier)
XGBClassifier is chosen for its high accuracy and performance on tabular data.
It's a gradient boosting method that builds multiple trees and combines them to minimize error.
The model is trained on the scaled training data.
7. Model Evaluation
The trained model is tested on the test set.
Evaluation metrics include:
Classification Report: Precision, recall, f1-score for each crop class.
Confusion Matrix: Helps visualize misclassifications between crops.
8. Saving the Model
The trained model, label encoder, and scaler are saved using joblib so they can be reused for predictions without retraining.

# Benefits of This Approach
Synthetic data helps balance the dataset and improve accuracy for underrepresented crops.
XGBoost provides fast training and high performance.
Scalable and reusable pipeline with saved models and encoders.

---

# Fertilizer Recommendation System Using Machine Learning
# Description
The Fertilizer Recommendation System is a machine learning-based application that provides recommendations for suitable crops based on various environmental and soil conditions. It aims to assist farmers and agricultural professionals in making informed decisions about fertilizer selection, optimizing yields, and maximizing profitability.

The system takes into account several factors such as temperature, humidity, moisture, nitrogen, phosphorus, potassium, soil type and crop type to determine the most suitable fertilizer for a given region. By analyzing historical data and using predictive models, the system provides personalized recommendations tailored to the specific conditions of a farm or agricultural area.

# Theoretical Overview
This system predicts the appropriate fertilizer for a given crop and soil condition using machine learning. It ensures label integrity, encodes categorical variables, scales numerical features, trains a Random Forest classifier, evaluates its accuracy, and saves all components for deployment.

1. Known Class Definitions
Predefined classes for crops, soil types, and fertilizers are specified to validate and restrict the model to recognized categories. This reduces the risk of training on incorrect or unknown labels.
2. Label Validation
Before proceeding with encoding, the dataset is checked for any labels not present in the known classes. This ensures data consistency and prevents runtime errors during model training.
3. Data Loading
The dataset is imported from a CSV file and stored in a structured format (DataFrame) for easy manipulation and processing.
4. Label Encoding
Categorical features such as crop and soil type, which are strings, are converted into numeric codes using LabelEncoder. This is essential because machine learning models require numerical inputs.
5. Feature and Target Separation
Input features (like temperature, moisture, nutrient levels, etc.) are separated from the target variable (fertilizer type). This distinction is necessary for supervised learning.
6. Train-Test Split
The dataset is divided into training and test sets. A stratified split ensures that the distribution of classes remains consistent across both sets, improving evaluation fairness.
7. Feature Scaling
Numerical features are standardized using StandardScaler, which adjusts them to have zero mean and unit variance. This step helps many models perform better and converge faster.
8. Model Training 
A Random Forest classifier is trained on the scaled training data. Random Forest is an ensemble method that combines multiple decision trees to reduce overfitting and improve accuracy.
9. Model Evaluation
The model is evaluated using test data, and metrics like accuracy and classification report (precision, recall, F1-score) are used to measure its performance.
10. Model and Encoder Saving
The trained model, along with the encoders and scaler, are saved to disk using joblib. This allows for the model to be reused without retraining, making it suitable for deployment.

---

# Plant Disease detection System Using Machine Learning
# Description
The Plant Disease Detection System is a machine learning-powered application designed to identify plant diseases from images and recommend appropriate treatments. It aims to support farmers, gardeners, and agricultural experts by enabling early diagnosis and effective disease management, ultimately improving crop health and agricultural productivity.

The system takes an image of a diseased plant leaf as input and uses advanced computer vision and deep learning techniques to detect the type of disease affecting the plant. Once identified, it provides personalized treatment recommendations, which may include suitable pesticides, fungicides, or organic remedies, along with preventive care tips. By leveraging a rich dataset of plant diseases and their visual symptoms, the system ensures accurate and timely interventions, helping reduce crop losses and improve yield quality.

# Theoretical Overview
The system is designed to automatically detect plant diseases from images using a Convolutional Neural Network (CNN). It follows a typical machine learning pipeline involving data preprocessing, model training, and prediction.

1. Image Preprocessing
The input data consists of images of healthy and diseased plant leaves, organized in folders by class.
Each image is resized to a standard size (224x224) for uniformity.
Pixel values are normalized (rescaled between 0 and 1) to improve training stability.
The dataset is split into training and validation sets to allow the model to learn patterns and then be evaluated on unseen data.
2. CNN Model Architecture
A deep CNN is built with multiple convolutional layers to extract spatial features (edges, textures, patterns) from leaf images.
Pooling layers reduce dimensionality and computation.
Fully connected layers at the end help make predictions based on extracted features.
Dropout layers are used to prevent overfitting.
3. Model Training
The model is compiled using the Adam optimizer and trained using categorical cross-entropy loss since it’s a multi-class classification task.
Training runs for one or more epochs where the model learns to associate image features with disease classes.
Accuracy and loss are monitored during training and validation to assess model performance.
4. Model Evaluation
After training, the model is evaluated on the validation set to determine its ability to generalize to new data.
Accuracy and loss graphs help visualize training trends and detect overfitting or underfitting.
5. Saving and Using the Model
The trained model is saved in .h5 format for future use.
A helper function is created to load and preprocess any new image.
The model predicts the class of a new leaf image and returns the most likely disease label using the saved class mappings.
6. Real-world Application
A farmer or agronomist can upload a photo of a diseased plant leaf.
The model processes the image and outputs the predicted disease.
This enables timely diagnosis and helps in recommending proper treatment.

---

# Key Features - OVERALL
1. Input Data Collection: The system allows users to input relevant data such as soil parameters, climate information, and geographic location.
2. Data Preprocessing: The input data is preprocessed to handle missing values, normalize or scale features, and transform categorical variables.
3. Machine Learning Models: Various machine learning algorithms are employed, including decision trees, random forests, support vector machines (SVM), and gradient boosting techniques, to build predictive models.
4. Model Training and Evaluation: The models are trained on historical data and evaluated using appropriate performance metrics to ensure accuracy and reliability.
    a. crop [Logistic regression: 0.91, Naive bayes: 0.995, Bagging: 0.9886, Light GBM: 0.9890, Decision Tree Classifier: 0.9848, Random Forest: 0.9917, Logistic Regression: 0.9435, Adaboost: 0.095, Gradient Boost: 0.981, XGBoost: 0.99] = used XGBoost with synthetic data
    b. fertilizer [Decision Tree(0.90,0.93), Naive Bayes(0.94,0.91), SVM(0.99,0.99), Logistic Regression(0.78,0.90), Random Forest(1.0,1.0)] (test, train) = random forest took for final model training
    c. plant disease [epoch-50 - 3hours to train model - validation accuracy: 0.95 validation loss: 0.22, trained more epoch so accuracy increases and the loss decreases - 0.98, 0.15]
5. Crop and Fertilizer Recommendation: Based on the trained models, the system recommends the most suitable crops for the given input parameters.
6. User-Friendly Interface: The system provides a user-friendly interface where users can easily input their data, view recommendations, and explore additional information.

---

# Technologies Used
Python: Programming language used for model development, data preprocessing, and web application development.
Scikit-learn: Machine learning library used for model training, evaluation, and prediction.
Pandas: Data manipulation library used for data preprocessing and analysis.
NumPy: Library for numerical computing used for handling arrays and mathematical operations.
Flask: Web framework used for building the user interface and handling HTTP requests.
HTML/CSS: Markup and styling languages used for designing the web interface.
JavaScript: Scripting language used for client-side interactions and enhancing the user interface.

---


# 🚀 Installation and Usage

## 1. ✅ Set Up Virtual Environment

```bash
python3 -m venv myenv
```

### 🔹 Activate Environment

- **On PowerShell:**

```bash
.\myenv\Scripts\Activate.ps1
```

- **On Command Prompt (cmd):**

```cmd
myenv\Scriptsctivate
```

## 2. 📥 Clone the Repository

```bash
git clone https://github.com/Shalini-A23/AgriAssist.git
cd AgriAssist
```

## 3. 📦 Install Required Dependencies

```bash
pip install -r requirements.txt
```

## 4. ▶️ Run the Application

```bash
python app.py
```

## 5. 🌐 Access the Application

Open your browser and go to:

```
http://localhost:5000
```

---

# Future Enhancements
Integration of real-time weather data to improve the accuracy of recommendations.
Incorporation of crop market prices and profitability analysis to assist farmers in making economically viable decisions.
Development of a mobile application for convenient access and usage on smartphones and tablets.
Integration of user feedback and data collection to continuously enhance the recommendation system's performance.

---

# CROP RECOMMENDATION
# CROPS WE CLASSIFIED IN TRAINING DATA AND LABELED FOR PREDICTION
1. rice
2. maize
3. jute
4. cotton
5. coconut
6. papaya
7. orange
8. apple
9. muskmelon
10. watermelon
11. grapes
12. mango
13. anana
14. pomegranate
15. lentil
16. blackgram
17. mungbean
18. mothbeans
19. pigeonpeas
20. kidneybeans
21. chickpea
22. coffee

# test cases 01:
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 27.0,
  "humidity": 80.0,
  "ph": 6.5,
  "rainfall": 200.0
}
=>RICE Prefers high humidity, warm temps, and high rainfall.

# test cases 02:
{
  "N": 80,
  "P": 55,
  "K": 60,
  "temperature": 24.5,
  "humidity": 65.0,
  "ph": 6.8,
  "rainfall": 85.0
}
=> MAIZE Grows well in moderate humidity and temperature.

# test cases 03:
{
  "N": 80,
  "P": 80,
  "K": 80,
  "temperature": 29,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 250
}
=> BANANA Needs high K, warm + humid climate, and good rainfall.

# test cases 04:
{
  "N": 80,
  "P": 40,
  "K": 40,
  "temperature": 29,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 250
}
=> COCONUT Needs warm, humid, and high-rainfall areas. 

# test cases 05:
{
  "N": 95,
  "P": 37,
  "K": 35,
  "temperature": 27,
  "humidity": 68,
  "ph": 6.3,
  "rainfall": 192
}
=> COFFEE Requires shade, humid, tropical climate, medium rainfall.

# test cases 06:
{
  "N": 120,
  "P": 48,
  "K": 16,
  "temperature": 22,
  "humidity": 75,
  "ph": 7.4,
  "rainfall": 71
}
=> COTTON Needs warm weather, moderate water, prefers well-drained soil.

# test cases 07:
{
  "N": 50,
  "P": 60,
  "K": 47,
  "temperature": 32,
  "humidity": 92,
  "ph": 6.9,
  "rainfall": 93
}
=> PAPAYA Hot + humid preferred, good NPK balance.

# test cases 08:
{
  "N": 35,
  "P": 128,
  "K": 205,
  "temperature": 21,
  "humidity": 93,
  "ph": 6.4,
  "rainfall": 107
}
=> APPLE Requires cool climate, high K, decent P. 

# test cases 09:
{
  "N": 40,
  "P": 68,
  "K": 17,
  "temperature": 34,
  "humidity": 65,
  "ph": 7.7,
  "rainfall": 70
}
=> BLACKGRAM Requires warm weather, low rainfall, less K.

# test cases 10:
{
  "N": 28,
  "P": 122,
  "K": 197,
  "temperature": 19,
  "humidity": 82,
  "ph": 5.8,
  "rainfall": 69
}
=> GRAPES Likes cool-to-warm climates, high P & K. 

---

# FERTILIZER RECOMMENDATION
# FERTILIZER WE CLASSIFIED IN TRAINING DATA AND LABELED FOR PREDICTION
1. 10-10-10
2. 10-26-26
3. 14-14-14
4. 14-35-14
5. 15-15-15
6. 17-17-17
7. 20-20
8. 28-28
9. DAP
10. Potassium Chloride
11. Potassium sulfate
12. Superphosphate
13. TSP
14. Urea

# input
temp, humi, moisture
n, p, k
soil type [black, clayey, loamy, red, sandy]
crop type [barley, cotton, ground nuts, maize, millets, oil seeds, paddy, pulses, sugarcane, tobacco, wheat, coffee, kidneybeans, orange, pomegranate, rice, watermelon]


# test case: 01
{
  "temperature": 26,
  "humidity": 53,
  "moisture": 28,
  "soil_type": "loamy",
  "crop_type": "coffee",
  "N": 85,
  "P": 33,
  "K": 25
}
=> UREA N-P-K Ratio: 46-0-0 — Fast nitrogen supply to leafy vegetables and nitrogen-deficient soil. NITROGEN-RICH FERTILIZER

# test case: 02
{
  "temperature": 31,
  "humidity": 62,
  "moisture": 49,
  "soil_type": "black",
  "crop_type": "sugarcane",
  "N": 10,
  "P": 13,
  "K": 14
}
=> 17-17-17 N-P-K Ratio: 17-17-17 — Strong multi-nutrient boost. BALANCED NPK FERTILIZER

# test case: 03
{
  "temperature": 30,
  "humidity": 83,
  "moisture": 63,
  "soil_type": "loamy",
  "crop_type": "pomegranate",
  "N": 39,
  "P": 30,
  "K": 38
}
=> Potassium sulfate N-P-K Ratio: 0-0-50 — Suitable for chloride-sensitive crops; provides sulfur. POTASSIUM-RICH FERTILIZER

# test case: 04
{
  "temperature": 25,
  "humidity": 78,
  "moisture": 40,
  "soil_type": "loamy",
  "crop_type": "cotton",
  "N": 102,
  "P": 37,
  "K": 25
}
=> DAP N-P-K Ratio: 18-46-0 — Supplies both nitrogen and phosphorus at planting. NITROGEN & PHOSPHORUS FERTILIZER

# test case: 05
{
  "temperature": 30,
  "humidity": 60,
  "moisture": 27,
  "soil_type": "red",
  "crop_type": "tobacco",
  "N": 4,
  "P": 17,
  "K": 17
}
=> 10-26-26 N-P-K Ratio: 10-26-26 — Boosts root and fruit development. PHOSPHORUS-RICH FERTILIZER

---

# MODEL SELECTION
# 🌾 1. Crop Recommendation – XGBoost
✅ Why XGBoost was used:
Crop recommendation involves complex patterns between soil, nutrients, and environmental features.
XGBoost handles non-linear relationships, missing values, and gives very high accuracy (99%), especially with synthetic + real data.
It is fast, efficient, and prevents overfitting with regularization.
⚙️ How XGBoost works:
General: An advanced boosting algorithm that builds models sequentially and corrects previous errors.
Technical: Uses gradient descent on loss function, adds trees one at a time, and combines them using weighted sum with L1/L2 regularization for accuracy and generalization.

# 🌱 2. Fertilizer Recommendation – Random Forest
✅ Why Random Forest was used:
Fertilizer prediction needs a model that handles both numerical and categorical data (soil type, crop type, nutrient level).
Random Forest provided perfect accuracy (1.0 test/train) in this case, making it the most stable and accurate choice.
It handles feature importance, is robust against overfitting, and works well on medium-sized tabular datasets.
⚙️ How Random Forest works:
General: An ensemble of many decision trees that vote for the final prediction.
Technical: Uses bagging (bootstrap aggregation), trains multiple trees on random subsets, and combines their results using majority voting for classification.

# 🍃 3. Plant Disease Detection – CNN (Convolutional Neural Network)
✅ Why CNN was used:
Plant disease detection is based on leaf images, where symptoms appear as spots, discoloration, or texture changes.
CNNs are designed to automatically extract features from images like edges, colors, and patterns.
After training with more epochs, the CNN achieved high validation accuracy (98%), making it ideal for this task.
⚙️ How CNN works:
General: A deep learning model that recognizes visual patterns in images using layers of filters.
Technical:
Convolution layers extract features (edges, shapes).
Pooling layers reduce dimensionality and retain important info.
Fully connected layers perform classification.
Trained with backpropagation and optimized using loss function.

---

## 🧠 ML Models Used and Accuracy Scores

| Task                      | Algorithms Tried                                                                                      | Final Model Used                 | Accuracy        |
|---------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------|------------------|
| **Crop Recommendation**   | Logistic Regression, Naive Bayes, Random Forest, Decision Tree, XGBoost, etc.                        | **XGBoost + Synthetic Data**     | **~99%**         |
| **Fertilizer Suggestion** | Decision Tree, Naive Bayes, SVM, Logistic Regression, Random Forest                                  | **Random Forest**                | **Train: 1.0**, **Test: ~0.93** |
| **Disease Detection**     | CNN (custom architecture)                                                                             | **CNN with 50+ epochs**          | **Val Acc: ~0.98**, **Loss: 0.15** |


---

# BASIC IDEA OF MACHINE LEARNING MODELS
# 🌾 Smart Agriculture System – Model Explanation

This document explains all the algorithms used for **crop prediction**, **fertilizer recommendation**, and **plant disease detection** in both simple and technical terms.

---

## 🅰️ Crop Recommendation

### ✅ Final Model: **XGBoost** (Accuracy: 0.99 with synthetic data)

### 🔹 Algorithms Tried

- Logistic Regression: 0.91
- Naive Bayes: 0.995
- Bagging: 0.9886
- LightGBM: 0.9890
- Decision Tree: 0.9848
- Random Forest: 0.9917
- Logistic Regression (tuned): 0.9435
- Adaboost: 0.095
- Gradient Boost: 0.981
- **XGBoost**: **0.99**

---

### 📘 Simple Explanation

| Algorithm | Simple Version |
|----------|----------------|
| Logistic Regression | Like a scale — checks features and tips toward one crop or another. |
| Naive Bayes | Guesses crop based on clues (e.g., "If soil is acidic, it’s probably rice"). |
| Decision Tree | Like a flowchart — “Is it hot? Yes → Is rainfall high? Yes → Grow rice!” |
| Random Forest | Like asking a group of trees to vote — more accurate. |
| Bagging | Builds many models on different data and averages them. |
| Adaboost | Each model learns from previous mistakes. |
| Gradient Boost | Improves by correcting errors gradually. |
| LightGBM | Very fast and accurate for structured data. |
| XGBoost | Powerful, smart ensemble of trees with regularization. |

---

### 🧠 Technical Explanation

| Algorithm | Working |
|----------|---------|
| Logistic Regression | Linear model using sigmoid function to output probabilities. |
| Naive Bayes | Probabilistic model assuming feature independence. |
| Decision Tree | Splits data by feature thresholds to maximize information gain. |
| Random Forest | Ensemble of Decision Trees with bagging. |
| Bagging | Bootstrap Aggregating of weak models. |
| Adaboost | Boosts weak learners by focusing on misclassified instances. |
| Gradient Boost | Adds models to correct residual errors step-by-step. |
| LightGBM | Gradient boosting with histogram learning and leaf-wise splits. |
| XGBoost | Regularized boosting with parallel computation and missing value handling. |

---

### ✅ Why XGBoost?

- Handles structured data well.
- Manages overfitting using regularization.
- Very high accuracy.
- Works well with synthetic data.

---

## 🅱️ Fertilizer Recommendation

### ✅ Final Model: **Random Forest** (Train/Test Accuracy: 1.0)

### 🔹 Algorithms Tried

- Decision Tree: (0.90, 0.93)
- Naive Bayes: (0.94, 0.91)
- SVM: (0.99, 0.99)
- Logistic Regression: (0.78, 0.90)
- **Random Forest**: (1.0, 1.0)

---

### 📘 Simple Explanation

| Algorithm | Simple Version |
|----------|----------------|
| Decision Tree | Asks questions: "Is nitrogen low? Is crop rice?" Then suggests fertilizer. |
| Naive Bayes | Makes guesses based on past patterns. |
| SVM | Draws invisible lines to separate fertilizers. |
| Logistic Regression | Predicts fertilizer like voting. |
| Random Forest | Group of decision trees that vote together. |

---

### 🧠 Technical Explanation

| Algorithm | Working |
|----------|---------|
| Decision Tree | Splits input features to reach a decision. |
| Naive Bayes | Uses probabilities and independence assumptions. |
| SVM | Finds optimal hyperplanes in high-dimensional space. |
| Logistic Regression | Uses sigmoid function to predict class probabilities. |
| Random Forest | Ensemble of trees that generalize well. |

---

### ✅ Why Random Forest?

- Accurate and stable.
- Handles many features well.
- Robust to outliers.
- Achieved perfect accuracy.

---

## 🅲️ Plant Disease Detection

### ✅ Final Model: **Convolutional Neural Network (CNN)**  
**Training Time**: 3 hours for 50 epochs  
**Validation Accuracy**: Increased from **0.95 → 0.98**  
**Validation Loss**: Decreased from **0.22 → 0.15**

---

### 📘 Simple Explanation

| Term | Meaning |
|------|---------|
| CNN | A smart system that "sees" leaf images and says if they’re sick or healthy. |
| Epoch | One full round of learning on all images. |
| Accuracy | How often it guesses right. |
| Loss | How wrong the model is — lower is better. |

---

### 🧠 Technical Explanation

- CNN is a deep learning model best for image inputs.
- It uses:
  - Convolution Layers (feature extraction)
  - Pooling Layers (downsampling)
  - Fully Connected Layers (classification)
  - Softmax (probability output)
- Learns patterns like spots, color changes on leaves.

---

### ✅ Why CNN?

- Only model type that understands images well.
- State-of-the-art performance on plant disease tasks.
- Achieved 98% accuracy after extended training.

---

## ✅ Summary Table

| Task | Final Model | Why Chosen |
|------|-------------|------------|
| Crop Recommendation | **XGBoost** | Excellent accuracy, robust to overfitting, handles structured data. |
| Fertilizer Prediction | **Random Forest** | Perfect accuracy, robust, interpretable. |
| Plant Disease Detection | **CNN** | Learns from images, highest performance for visual tasks. |

---
