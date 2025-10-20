
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load all models and components
print("Loading models and components...")
try:
    with open('./models/linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    with open('./models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('./models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('./models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('./models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    with open('./models/model_results.pkl', 'rb') as f:
        model_results = pickle.load(f)

    with open('./models/best_model_info.pkl', 'rb') as f:
        best_model_info = pickle.load(f)

    print("✓ All models loaded successfully!")

except FileNotFoundError as e:
    print(f"⚠️ Model file not found: {e.filename}")
    raise
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    raise

# Create comparison visualization
def create_comparison_chart():
    try:
        models = list(model_results.keys())
        r2_scores = [model_results[model]['R²'] for model in models]
        rmse_scores = [model_results[model]['RMSE'] for model in models]
        mae_scores = [model_results[model]['MAE'] for model in models]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('R² Score (Accuracy)', 'RMSE (Lower is Better)', 'MAE (Lower is Better)'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # R² Score chart
        colors_r2 = ['gold' if model == best_model_info['best_model_name'] else 'lightblue' for model in models]
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score', marker_color=colors_r2, showlegend=False),
            row=1, col=1
        )
        
        # RMSE chart
        colors_rmse = ['gold' if model == best_model_info['best_model_name'] else 'lightcoral' for model in models]
        fig.add_trace(
            go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color=colors_rmse, showlegend=False),
            row=1, col=2
        )
        
        # MAE chart
        colors_mae = ['gold' if model == best_model_info['best_model_name'] else 'lightgreen' for model in models]
        fig.add_trace(
            go.Bar(x=models, y=mae_scores, name='MAE', marker_color=colors_mae, showlegend=False),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Model Performance Comparison",
            title_x=0.5,
            title_font_size=20
        )
        
        fig.update_xaxes(tickangle=0)
        
        return fig
    except KeyError as e:
        print(f"Missing key in model results: {e}")
        return None
    except Exception as e:
        print(f"Error creating comparison chart: {e}")
        return None

# Function to get model comparison table
def get_model_comparison():
    comparison_data = []
    for model_name, metrics in model_results.items():
        is_best = "⭐ BEST" if model_name == best_model_info['best_model_name'] else ""
        comparison_data.append({
            'Model': f"{model_name} {is_best}",
            'R² Score': f"{metrics['R²']:.4f}",
            'RMSE': f"{metrics['RMSE']:.2f}",
            'MAE': f"{metrics['MAE']:.2f}",
            'MSE': f"{metrics['MSE']:.2f}"
        })
    
    return pd.DataFrame(comparison_data)

# Function to explain why best model was chosen
def get_best_model_explanation():
    best_name = best_model_info['best_model_name']
    best_r2 = best_model_info['best_r2']
    best_rmse = best_model_info['best_rmse']
    
    explanation = f"""
    ## 🏆 Best Model Selection: **{best_name}**
    
    ### Why {best_name} was chosen:
    
    **1. Highest R² Score: {best_r2:.4f}**
    - R² score measures how well the model explains the variance in the data
    - Score closer to 1.0 indicates better predictions
    - {best_name} achieved the highest R² among all models
    
    **2. Lowest RMSE: {best_rmse:.2f} calories**
    - RMSE (Root Mean Squared Error) measures prediction accuracy
    - Lower RMSE means predictions are closer to actual values
    - {best_name} has the smallest prediction error
    
    **3. Model Characteristics:**
    """
    
    if best_name == "Linear Regression":
        explanation += """
    - **Interpretability**: Easy to understand and explain
    - **Speed**: Fast training and prediction
    - **Simplicity**: Works well when relationships are linear
        """
    elif best_name == "Random Forest":
        explanation += """
    - **Robustness**: Handles non-linear relationships well
    - **Feature Importance**: Provides insights into important features
    - **Accuracy**: Ensemble method reduces overfitting
    - **Versatility**: Works well with various types of data
        """
    else:  # XGBoost
        explanation += """
    - **Performance**: State-of-the-art gradient boosting algorithm
    - **Accuracy**: Excellent predictive power
    - **Handling Complex Patterns**: Captures intricate relationships
    - **Industry Standard**: Widely used in competitions and production
        """
    
    explanation += f"""
    
    ### 📊 Performance Summary:
    - **Accuracy (R²)**: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)
    - **Error (RMSE)**: {best_rmse:.2f} calories
    - **Reliability**: Consistently outperforms other models
    """
    
    return explanation

# Prediction function
def predict_calories(*inputs):
    try:
        # Create input dataframe
        input_dict = {feature: [value] for feature, value in zip(feature_names, inputs)}
        input_df = pd.DataFrame(input_dict)
        
        # Get predictions from all models
        # Linear Regression needs scaled input
        input_scaled = scaler.transform(input_df)
        lr_pred = lr_model.predict(input_scaled)[0]
        
        # Random Forest and XGBoost use unscaled input
        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]
        
        # Create results
        results_text = f"""
        ### Prediction Results:
        
        **Linear Regression**: {lr_pred:.2f} calories
        **Random Forest**: {rf_pred:.2f} calories  
        **XGBoost**: {xgb_pred:.2f} calories
        
        ### 🎯 Best Model Prediction ({best_model_info['best_model_name']}):
        """
        
        if best_model_info['best_model_name'] == 'Linear Regression':
            results_text += f"**{lr_pred:.2f} calories**"
            best_pred = lr_pred
        elif best_model_info['best_model_name'] == 'Random Forest':
            results_text += f"**{rf_pred:.2f} calories**"
            best_pred = rf_pred
        else:
            results_text += f"**{xgb_pred:.2f} calories**"
            best_pred = xgb_pred
        
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Linear Regression', 'Random Forest', 'XGBoost'],
                y=[lr_pred, rf_pred, xgb_pred],
                marker_color=['lightblue', 'lightgreen', 'coral'],
                text=[f'{lr_pred:.1f}', f'{rf_pred:.1f}', f'{xgb_pred:.1f}'],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Calorie Predictions by Model',
            yaxis_title='Predicted Calories',
            height=400
        )
        
        return results_text, fig
        
    except Exception as e:
        return f"Error in prediction: {str(e)}", None

# Build Gradio Interface
with gr.Blocks(title="Calorie Prediction Model Comparison", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔥 Calorie Prediction System")
    gr.Markdown("### Machine Learning Model Comparison & Prediction Tool")
    
    with gr.Tabs():
        # Tab 1: Model Comparison
        with gr.Tab("📊 Model Comparison"):
            gr.Markdown("## Model Performance Metrics")
            
            comparison_table = gr.Dataframe(
                value=get_model_comparison(),
                label="Model Comparison Table"
            )
            
            comparison_plot = gr.Plot(
                value=create_comparison_chart(),
                label="Performance Visualization"
            )
            
            gr.Markdown(get_best_model_explanation())
        
        # Tab 2: Make Predictions
        with gr.Tab("🎯 Make Predictions"):
            gr.Markdown("## Enter Values to Predict Calories Burned")
            
            # Create input components dynamically based on features
            inputs = []
            with gr.Row():
                with gr.Column():
                    mid_point = len(feature_names) // 2
                    for feature in feature_names[:mid_point]:
                        inputs.append(gr.Number(label=feature, value=0))
                with gr.Column():
                    for feature in feature_names[mid_point:]:
                        inputs.append(gr.Number(label=feature, value=0))
            
            predict_btn = gr.Button("🔮 Predict Calories", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    prediction_output = gr.Markdown()
                with gr.Column():
                    prediction_chart = gr.Plot()
            
            predict_btn.click(
                fn=predict_calories,
                inputs=inputs,
                outputs=[prediction_output, prediction_chart]
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Application
            
            This application demonstrates a comparison of three machine learning models for predicting calories burned during exercise:
            
            ### Models Implemented:
            1. **Linear Regression**: A simple, interpretable baseline model
            2. **Random Forest**: An ensemble method using multiple decision trees
            3. **XGBoost**: Advanced gradient boosting algorithm
            
            ### Evaluation Metrics:
            - **R² Score**: Measures how well the model explains variance (higher is better)
            - **RMSE**: Root Mean Squared Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            
            ### Dataset:
            - Features include exercise and physical characteristics
            - Target variable: Calories burned
            
            ### How to Use:
            1. View model comparison in the first tab
            2. Make predictions using the second tab
            3. The best model is automatically selected based on performance
            
            ---
            *Built with Gradio, scikit-learn, XGBoost, and Plotly*
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch()