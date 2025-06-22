import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("Disease Prediction with Explainable AI (SHAP)")

# Load model and scaler
@st.cache_resource
def load_models():
    model = joblib.load('C:/Users/Kavya/OneDrive/Desktop/Contributions/Data_Scientist/DataSentience-AIML/Health AI/DiseasePrediction_XAI/models/random_forest_model.pkl')
    scaler = joblib.load('C:/Users/Kavya/OneDrive/Desktop/Contributions/Data_Scientist/DataSentience-AIML/Health AI/DiseasePrediction_XAI/models/scaler.pkl')
    return model, scaler

model, scaler = load_models()

# Feature list
feature_names = ['Age', 'Gender', 'Comorbidity', 'Coronary Artery Disease (CAD)', 'Hypothyroidism',
                'Hyperlipidemia', 'Diabetes Mellitus (DM)', 'Height', 'Weight', 'Body Mass Index (BMI)', 
                'Total Body Water (TBW)', 'Extracellular Water (ECW)', 'Intracellular Water (ICW)', 
                'Extracellular Fluid/Total Body Water (ECF/TBW)', 'Total Body Fat Ratio (TBFR) (%)', 
                'Lean Mass (LM) (%)', 'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)', 
                'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)', 'Total Fat Content (TFC)', 
                'Visceral Fat Area (VFA)', 'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)', 
                'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)', 
                'Triglyceride', 'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)', 
                'Alkaline Phosphatase (ALP)', 'Creatinine', 'Glomerular Filtration Rate (GFR)', 
                'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D']

# Input form
col1, col2 = st.columns(2)
input_data = []
with col1:
    for feature in feature_names[:len(feature_names)//2]:
        value = st.number_input(f"{feature}", value=0.0, step=0.1)
        input_data.append(value)
with col2:
    for feature in feature_names[len(feature_names)//2:]:
        value = st.number_input(f"{feature}", value=0.0, step=0.1)
        input_data.append(value)

# Prepare input
input_array = np.array(input_data).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=feature_names)
input_scaled = scaler.transform(input_df)

# Custom function for SHAP HTML plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

if st.button('Predict Disease', type="primary"):
    with st.spinner('Analyzing...'):
        try:
            # Prediction
            prediction = model.predict(input_scaled)
            pred_proba = model.predict_proba(input_scaled)[0]
            
            st.success(f"## Prediction: {'Disease' if prediction[0] == 1 else 'No Disease'}")
            st.write(f"**Confidence**: {max(pred_proba)*100:.1f}%")

            # SHAP explanation using scaled input
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)

            # Debug information
            st.write(f"SHAP values type: {type(shap_values)}")
            st.write(f"SHAP values shape: {np.array(shap_values).shape}")

            # Handle different SHAP return formats
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_vals_positive = shap_values[1][0]  # Class 1, first sample
                    base_val = explainer.expected_value[1]
                else:
                    shap_vals_positive = shap_values[0][0]
                    base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                if len(shap_values.shape) == 3:
                    shap_vals_positive = shap_values[0, :, 1]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_vals_positive = shap_values[0]
                    base_val = explainer.expected_value

            st.write(f"Using SHAP values shape: {shap_vals_positive.shape}")
            st.write(f"Using base value: {base_val}")

            # SHAP Visualizations
            tab1, tab2, tab3 = st.tabs(["Waterfall Plot", "Force Plot", "Feature Importance"])

            # Waterfall Plot
            with tab1:
                st.subheader("Waterfall Plot")
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    explanation = shap.Explanation(
                        values=shap_vals_positive,
                        base_values=base_val,
                        data=input_df.iloc[0].values,
                        feature_names=feature_names
                    )
                    
                    shap.plots.waterfall(explanation, max_display=15, show=False)
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Waterfall plot error: {str(e)}")

            # Force Plot - CORRECTED VERSION
            with tab2:
                st.subheader("Force Plot")
                
                # Method 1: Try interactive JavaScript version first
                try:
                    st.write("**Interactive Force Plot:**")
                    force_plot = shap.force_plot(
                        base_val,
                        shap_vals_positive,
                        input_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False  # Use interactive version
                    )
                    st_shap(force_plot, height=400)
                    
                except Exception as e:
                    st.error(f"Interactive force plot error: {str(e)}")
                    
                    # Method 2: Custom Force Plot Implementation
                    try:
                        st.write("**Custom Force Plot Visualization:**")
                        fig, ax = plt.subplots(figsize=(16, 6))
                        
                        # Calculate cumulative effects
                        sorted_idx = np.argsort(np.abs(shap_vals_positive))[::-1][:15]
                        sorted_shap = shap_vals_positive[sorted_idx]
                        sorted_names = [feature_names[i] for i in sorted_idx]
                        sorted_values = input_df.iloc[0].values[sorted_idx]
                        
                        # Create force plot visualization
                        cumulative = base_val
                        positions = [cumulative]
                        
                        for shap_val in sorted_shap:
                            cumulative += shap_val
                            positions.append(cumulative)
                        
                        # Draw the force plot
                        for i, (name, shap_val, feat_val) in enumerate(zip(sorted_names, sorted_shap, sorted_values)):
                            color = '#ff6b6b' if shap_val > 0 else '#4ecdc4'
                            
                            # Draw arrow
                            ax.arrow(positions[i], 0.5, shap_val, 0, 
                                    head_width=0.05, head_length=abs(shap_val)*0.05, 
                                    fc=color, ec=color, alpha=0.8, width=0.02)
                            
                            # Add text labels
                            text_x = positions[i] + shap_val/2
                            text_y = 0.6 if shap_val > 0 else 0.4
                            
                            ax.text(text_x, text_y, f'{name}\n{feat_val:.2f}\nSHAP: {shap_val:.3f}', 
                                   ha='center', va='center', fontsize=8, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
                        
                        # Add base value and prediction lines
                        ax.axvline(x=base_val, color='gray', linestyle='--', alpha=0.7, 
                                  label=f'Base Value: {base_val:.3f}')
                        ax.axvline(x=positions[-1], color='black', linestyle='-', alpha=0.9, 
                                  label=f'Prediction: {positions[-1]:.3f}')
                        
                        ax.set_ylim(0, 1)
                        ax.set_xlim(min(base_val, positions[-1]) - 0.1, 
                                   max(base_val, positions[-1]) + 0.1)
                        ax.set_xlabel('Model Output')
                        ax.set_title('SHAP Force Plot - Feature Contributions to Disease Prediction')
                        ax.legend()
                        ax.grid(axis='x', alpha=0.3)
                        
                        # Remove y-axis ticks and labels
                        ax.set_yticks([])
                        
                        plt.tight_layout()
                        st.pyplot(fig, bbox_inches='tight')
                        plt.close()
                        
                    except Exception as e2:
                        st.error(f"Custom force plot failed: {str(e2)}")
                        
                        # Method 3: Fallback - Enhanced Bar Plot
                        try:
                            st.write("**Feature Impact Visualization (Fallback):**")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Top 15 features by absolute SHAP value
                            top_idx = np.argsort(np.abs(shap_vals_positive))[::-1][:15]
                            top_shap = shap_vals_positive[top_idx]
                            top_names = [feature_names[i] for i in top_idx]
                            top_values = input_df.iloc[0].values[top_idx]
                            
                            # Create horizontal bar plot
                            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in top_shap]
                            bars = ax.barh(range(len(top_shap)), top_shap, color=colors, alpha=0.8)
                            
                            # Customize plot
                            ax.set_yticks(range(len(top_shap)))
                            ax.set_yticklabels([f'{name}\n(value: {val:.2f})' for name, val in zip(top_names, top_values)])
                            ax.set_xlabel('SHAP Value (Impact on Disease Prediction)')
                            ax.set_title(f'Feature Contributions to Disease Prediction\nBase Value: {base_val:.3f} → Final Prediction: {base_val + np.sum(shap_vals_positive):.3f}')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add value labels
                            for i, (bar, val) in enumerate(zip(bars, top_shap)):
                                ax.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                                       va='center', ha='left' if val >= 0 else 'right', fontsize=9)
                            
                            # Add explanation
                            ax.text(0.02, 0.98, 'Red: Increases Disease Risk\nBlue: Decreases Disease Risk', 
                                   transform=ax.transAxes, va='top', ha='left',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                            
                            plt.tight_layout()
                            st.pyplot(fig, bbox_inches='tight')
                            plt.close()
                            st.info("Showing enhanced feature impact visualization as force plot alternative")
                            
                        except Exception as e3:
                            st.error(f"All force plot methods failed: {str(e3)}")

            # Feature Importance
            with tab3:
                st.subheader("Feature Importance")
                try:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    importance_values = np.abs(shap_vals_positive)
                    sorted_idx = np.argsort(importance_values)[::-1][:20]
                    
                    bars = ax.barh(range(len(sorted_idx)), importance_values[sorted_idx], 
                                  color='#2E86AB', alpha=0.8)
                    
                    ax.set_yticks(range(len(sorted_idx)))
                    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
                    ax.set_xlabel('|SHAP Value|')
                    ax.set_title('Feature Importance for Disease Prediction\n(Based on Absolute SHAP Values)')
                    
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=9)
                    
                    ax.grid(axis='x', alpha=0.3)
                    ax.set_axisbelow(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close()
                    
                    # Feature importance table
                    st.subheader("Top 10 Most Important Features")
                    importance_df = pd.DataFrame({
                        'Feature': [feature_names[i] for i in sorted_idx[:10]],
                        'Importance': importance_values[sorted_idx[:10]]
                    })
                    st.dataframe(importance_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Feature importance error: {str(e)}")

            # Interpretation guide
            with st.expander("📖 How to interpret these results"):
                st.markdown("""
                ### Understanding SHAP Visualizations:
                
                **🔄 Waterfall Plot:**
                - Shows step-by-step how each feature contributes to the final prediction
                - **Red bars**: Features that increase disease probability
                - **Blue bars**: Features that decrease disease probability
                
                **⚡ Force Plot:**
                - Visual representation of how features "push" the prediction from base value to final prediction
                - **Red arrows/bars**: Features pushing towards higher disease probability
                - **Blue arrows/bars**: Features pushing towards lower disease probability
                - The final prediction is the sum of base value + all SHAP contributions
                
                **📊 Feature Importance:**
                - Shows which features have the biggest absolute impact
                - Higher bars = more influential features for this prediction
                
                **💡 Key Points:**
                - SHAP values represent marginal contributions to the prediction
                - Positive SHAP = increases disease probability
                - Negative SHAP = decreases disease probability
                - Sum of all SHAP values + base value = final prediction score
                """)

        except Exception as e:
            st.error(f"Error during SHAP analysis: {str(e)}")
            st.write(f"SHAP version: {shap.__version__}")
            
            import traceback
            st.code(traceback.format_exc())
