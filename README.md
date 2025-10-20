# 🔥 Calorie Burn Prediction System

A machine learning web application that predicts calories burned during exercise using multiple ML models with an interactive Gradio interface.

## 🚀 Features

- **Multiple ML Models**: Compare Linear Regression, Random Forest, and XGBoost
- **Interactive Web Interface**: User-friendly Gradio app with tabs for comparison and prediction
- **Model Performance Visualization**: Charts showing R², RMSE, and MAE metrics
- **Real-time Predictions**: Get instant calorie burn predictions with visual comparisons
- **Best Model Selection**: Automatically identifies and highlights the best performing model

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd calorie-burned-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. Ensure trained models are in the `models/` directory
2. Run the application:
```bash
python app.py
```
3. Open your browser and navigate to the provided local URL
4. Use the interface to:
   - View model performance comparisons
   - Make calorie burn predictions

## 📊 Model Performance

The application compares three machine learning models:

- **Linear Regression**: Fast and interpretable
- **Random Forest**: Handles non-linear relationships
- **XGBoost**: High-performance gradient boosting

Performance metrics include:
- R² Score (accuracy)
- RMSE (prediction error)
- MAE (mean absolute error)

## 📁 Project Structure

```
calorie-burned-prediction/
├── models/                 # Trained ML models and components
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── model_results.pkl
│   └── best_model_info.pkl
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Dependencies

- `gradio==4.44.0` - Web interface framework
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.2` - Data manipulation
- `scikit-learn==1.5.2` - Machine learning library
- `xgboost==2.1.1` - Gradient boosting framework
- `plotly==5.17.0` - Interactive visualizations

## 🚨 Error Handling

The application includes comprehensive error handling for:
- Missing model files
- Corrupted pickle files
- Prediction errors
- Invalid input data

## 📝 License

This project is open source and available under the MIT License.