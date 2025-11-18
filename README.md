# 🫀 Cardiovascular Monitor

A professional clinical decision support application for heart disease risk assessment using machine learning. This Streamlit-based tool predicts cardiovascular disease risk and provides evidence-based medical recommendations.

## ✨ Features

- **Real-time Risk Prediction** - Instant heart disease risk assessment using XGBoost classifier
- **Individual Predictions** - Input patient vitals and get risk probability with visual gauge meter
- **Batch Predictions** - Upload CSV file with multiple patients for bulk analysis
- **Medical Insights** - Structured, evidence-based recommendations based on patient data
- **Prediction History** - Track all predictions with filtering by risk level and date range
- **Data Visualization** - Interactive charts comparing patient values against healthy/diseased averages

## 🔬 Model Information

**Best Performing Model: XGBoost Classifier**

| Metric | Value |
|--------|-------|
| Accuracy | 88.59% |
| Precision | 91.35% |
| Recall | 88.79% |
| F1-Score | 0.9005 |


**Features Used:**
- Age, Sex, Chest Pain Type, Resting Blood Pressure
- Cholesterol, Fasting Blood Sugar
- Resting Electrocardiographic Results, Maximum Heart Rate
- Exercise Induced Angina, Oldpeak
- ST Slope

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
\`\`\`bash
git clone https://github.com/yourusername/cardiovascular-monitor.git
cd cardiovascular-monitor
\`\`\`

2. **Create virtual environment**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. **Install dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. **Run the app**
\`\`\`bash
streamlit run cardiovascular_monitor.py
\`\`\`

5. **Open in browser**
\`\`\`
http://localhost:8501
\`\`\`

### Cloud Deployment

The app is deployed on Streamlit Cloud and publicly accessible at:
\`\`\`
https://cardiovascular-monitor.streamlit.app
\`\`\`

## 📋 Files in This Repository

\`\`\`
cardiovascular-monitor/
├── cardiovascular_monitor.py         # Main Streamlit application
├── best_heart_model.pkl              # Trained XGBoost model
├── preprocessor.pkl                  # Data preprocessor (StandardScaler)
├── heart.csv                         # Training dataset
├── model_performance.csv             # Model evaluation metrics
├── train_models.py                   # Model training script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
\`\`\`

## 💻 Usage Guide

### Single Patient Prediction

1. Navigate to the **"Predict"** tab
2. Enter patient clinical data:
   - Age (years)
   - Sex (Male/Female)
   - Chest Pain Type
   - Blood Pressure, Cholesterol, etc.
3. System validates inputs for physiological accuracy
4. Click **"Perform Assessment"** button
5. View results with:
   - Risk probability (%)
   - Risk gauge meter (Green/Yellow/Red)
   - Comparison charts
   - Medical recommendations

### Batch Predictions

1. Navigate to **"Batch Prediction"** tab
2. Prepare CSV with columns matching the expected format
3. Upload CSV file
4. System processes all records automatically
5. Download results as CSV with predictions

### View Prediction History

1. Go to **"Prediction History"** tab
2. Filter by:
   - Risk Category (Low/Medium/High)
   - Probability Threshold
3. View all previous predictions
4. Download filtered results

## 🔧 Technical Stack

- **Frontend:** Streamlit (Python web framework)
- **ML Model:** XGBoost Classifier
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Preprocessing:** StandardScaler
- **Deployment:** Streamlit Cloud

## 📊 Model Training

To retrain the model with new data:

\`\`\`bash
python train_models.py
\`\`\`

This will:
- Load and preprocess heart.csv
- Train 6 different models
- Select the best model (XGBoost)
- Save model, preprocessor, and metrics

## ⚠️ Medical Disclaimer

This application is a **decision-support tool** and NOT a medical device. All predictions and recommendations must be reviewed by qualified **healthcare professionals.** Do not rely solely on this tool for medical decisions. Always consult with a physician for proper diagnosis and treatment planning.

## 🔐 Data Privacy

- All predictions are processed locally
- No data is stored on external servers
- Users' input data is not retained after predictions

## 📈 Performance Insights

The application compares 6 machine learning models:

1. **XGBoost** - 88.59% ⭐ Selected
2. Gradient Boosting - 88.04%
3. Random Forest - 87.50%
4. Support Vector Machine - 86.41%
5. Logistic Regression - 85.33%
6. Decision Tree - 79.35%

## 🛠️ Dependencies

\`\`\`
streamlit
pandas
numpy
joblib
scikit-learn
xgboost
plotly
matplotlib
\`\`\`

## 📝 Input Validation

The app includes intelligent validation:
- Age must be between 20-100
- Maximum Heart Rate should be reasonable for age
- Cholesterol must be > 100
- Blood Pressure must be > 60
- Invalid inputs prevent prediction

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Contact: [hamza.gcuf.edu@gmail.com]

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author
**[Hamza Haleem]**
- LinkedIn: [Hamza Haleem](https://www.linkedin.com/in/hamza-analyst)

Created as a clinical decision support application for heart disease risk assessment.

---

**🔗 Live Demo:** https://cardiovascular-monitor.streamlit.app
