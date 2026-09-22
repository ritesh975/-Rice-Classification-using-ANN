# -Rice-Classification-using-ANN
Rice Classification using Artificial Neural Network (ANN)
<div align="center">

# 🌾 Rice Classification using ANN

### Binary Rice Classification with Deep Learning & Streamlit

![3D Animation](rice-ann-3d.gif)

**Artificial Neural Network • TensorFlow/Keras • Streamlit • Python**

<p>
  <a href="YOUR_STREAMLIT_DEPLOYMENT_LINK">🚀 Live Demo</a>
  &nbsp;•&nbsp;
  <a href="YOUR_GITHUB_REPOSITORY_LINK">💻 GitHub Repository</a>
</p>

</div>

---

## 📌 About the Project

**Rice Classification using ANN** is a Deep Learning project that uses an **Artificial Neural Network (ANN)** to perform **binary classification** of rice samples.

The model learns from numerical characteristics of rice grains and predicts the corresponding **Class 0 or Class 1**.

The trained model is integrated into an interactive **Streamlit web application**, making the project suitable for real-time prediction and deployment.

---

## ✨ Highlights

- 🧠 Artificial Neural Network based classification
- 📊 Numerical feature preprocessing
- ⚖️ StandardScaler for feature scaling
- 🔥 ReLU activation in hidden layers
- 🎯 Sigmoid activation for binary output
- 📉 Binary Crossentropy loss
- ⚡ Adam optimizer
- 📈 Accuracy and loss monitoring
- 🧩 Confusion Matrix & Classification Report
- 🖥️ Interactive Streamlit UI
- ☁️ Deployment-ready project

---

## 🧠 ANN Architecture

```text
                    INPUT FEATURES
                         │
                         ▼
                ┌─────────────────┐
                │  Dense Layer 64 │
                │      ReLU       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Dense Layer 32 │
                │      ReLU       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Dense Layer 16 │
                │      ReLU       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Output Layer 1  │
                │     Sigmoid     │
                └────────┬────────┘
                         │
                         ▼
                  CLASS 0 / CLASS 1
```

---

## 📊 Features Used

The ANN uses these numerical rice-grain characteristics:

| # | Feature |
|---:|---|
| 1 | Area |
| 2 | Major Axis Length |
| 3 | Minor Axis Length |
| 4 | Eccentricity |
| 5 | Convex Area |
| 6 | Equivalent Diameter |
| 7 | Extent |
| 8 | Perimeter |
| 9 | Roundness |
| 10 | Aspect Ratio |

---

## ⚙️ Model Configuration

| Parameter | Configuration |
|---|---|
| Problem Type | Binary Classification |
| Algorithm | Artificial Neural Network |
| Hidden Layers | 3 |
| Neurons | 64 → 32 → 16 |
| Hidden Activation | ReLU |
| Output Neurons | 1 |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Feature Scaling | StandardScaler |
| Framework | TensorFlow / Keras |

---

## 🔄 Machine Learning Workflow

```text
        📂 Rice Dataset
              │
              ▼
      🔍 Data Understanding
              │
              ▼
       🧹 Data Preprocessing
              │
              ▼
        ✂️ X / y Separation
              │
              ▼
       🔀 Train-Test Split
              │
              ▼
       ⚖️ StandardScaler
              │
              ▼
       🧠 ANN Model Training
              │
              ▼
         🎯 Prediction
              │
              ▼
      📈 Model Evaluation
              │
              ▼
       🖥️ Streamlit UI
              │
              ▼
        ☁️ Deployment
```

---

## 📈 Model Evaluation

The model performance can be evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Training Accuracy**
- **Validation Accuracy**
- **Training Loss**
- **Validation Loss**

---

## 🖥️ Streamlit Application

The Streamlit application allows users to enter rice-grain features and receive a prediction from the trained ANN model.

### Prediction Flow

```text
User Input
    ↓
Feature Scaling
    ↓
Trained ANN
    ↓
Sigmoid Probability
    ↓
0.5 Threshold
    ↓
Class 0 / Class 1
```

---

## 📁 Project Structure

```text
rice-classification-ann/
│
├── app.py                    # Streamlit application
├── model.keras               # Trained ANN model
├── scaler.pkl                # Saved StandardScaler
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── rice-ann-3d.gif           # Animated GitHub banner
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd rice-classification-ann
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## ☁️ Deployment

This project is ready for deployment using **Streamlit Community Cloud**.

### Deployment Steps

1. Push all project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select `app.py` as the main file.
6. Click **Deploy**.

### 🔗 Project Links

**Live Demo:**  
`YOUR_STREAMLIT_DEPLOYMENT_LINK`

**GitHub Repository:**  
`YOUR_GITHUB_REPOSITORY_LINK`

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🧠 TensorFlow/Keras | Deep Learning |
| 📊 Pandas | Data Processing |
| 🔢 NumPy | Numerical Computing |
| ⚙️ Scikit-learn | Scaling & Evaluation |
| 💾 Joblib | Scaler Serialization |
| 🖥️ Streamlit | Web UI & Deployment |

</div>

---

## 🔮 Future Improvements

- [ ] Multi-class rice classification
- [ ] Improved ANN architecture
- [ ] Interactive prediction probability chart
- [ ] Better UI/UX
- [ ] Model performance dashboard
- [ ] Additional rice varieties
- [ ] Cloud deployment

---

## 👨‍💻 Author

### **Your Name**

**MCA Student | Data Analytics & Deep Learning Enthusiast**

- 💻 GitHub: `YOUR_GITHUB_PROFILE_LINK`
- 🔗 LinkedIn: `YOUR_LINKEDIN_LINK`

---

## ⭐ Support

If you found this project useful, please consider giving the repository a **⭐ Star**.

<div align="center">

### 🌾 Rice Classification using Deep Learning

**Built with Python • TensorFlow • Keras • ANN • Streamlit**

</div>

