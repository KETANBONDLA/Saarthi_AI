import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and preprocessor
@st.cache_resource
def load_model():
    import os
    model_path = os.path.join('yield_prediction', 'dtrt.pkl')
    preprocessor_path = os.path.join('yield_prediction', 'preprocessor.pkl')
    model = pickle.load(open(model_path, 'rb'))
    preprocessor = pickle.load(open(preprocessor_path, 'rb'))
    return model, preprocessor

# Load the dataset for visualization and reference
@st.cache_data
def load_data():
    import os
    data_path = os.path.join('yield_prediction', 'data', 'yield_df.csv')
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    return df

# Function to make prediction
def predict_yield(model, preprocessor, year, rainfall, pesticides, temp, area, item):
    features = np.array([year, rainfall, pesticides, temp, area, item], dtype=object)
    transform_features = preprocessor.transform([features])
    predicted_yield = model.predict(transform_features).reshape(-1, 1)
    return predicted_yield[0][0]

# Add custom styling for better visuals
st.markdown("""
<style>
    .yield-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 1rem;
        text-align: center;
    }
    .yield-subheader {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2E7D32;
        margin: 1.5rem 0 1rem 0;
    }
    .yield-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 5px;
    }
    .viz-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .st-ae {
        font-size: 15px !important;
    }
    /* Override for smaller visualization font sizes */
    .viz-container .matplotlib-figure {
        font-size: 10px !important;
    }
    /* Compact table style */
    .compact-table {
        font-size: 0.85rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background-color: #e8f5e9;
        border-radius: 10px;
        border-left: 6px solid #2E7D32;
    }
    .info-text {
        font-size: 1rem;
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main function
def run():
    # Load model and data
    try:
        model, preprocessor = load_model()
        df = load_data()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        model_loaded = False
        df = None
    
    # App title and description
    st.markdown("<h1 class='yield-header'>🌾 Crop Yield Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 25px; font-size: 1.1rem;'>
        Predict crop yield (in hectograms per hectare) based on various environmental and agricultural factors.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for prediction and data insights
    tab1, tab2 = st.tabs(["Yield Prediction", "Data Insights"])
    
    with tab1:
        if model_loaded:
            st.markdown("<h2 class='yield-subheader'>Predict Crop Yield</h2>", unsafe_allow_html=True)
            
            # Create a form for input parameters
            with st.form("prediction_form"):
                st.markdown("<div class='yield-card'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get unique values for categorical features
                    areas = sorted(df['Area'].unique().tolist())
                    items = sorted(df['Item'].unique().tolist())
                    
                    # Year input
                    year = st.number_input("Year", min_value=1990, max_value=2030, value=2023)
                    
                    # Rainfall input
                    rainfall = st.number_input("Average Rainfall (mm per year)", 
                                            min_value=0.0, 
                                            max_value=5000.0, 
                                            value=1000.0,
                                            step=10.0)
                    
                    # Area (country) selection
                    area = st.selectbox("Area/Country", areas)
                
                with col2:
                    # Pesticides input
                    pesticides = st.number_input("Pesticides (tonnes)", 
                                                min_value=0.0, 
                                                max_value=10000.0, 
                                                value=100.0,
                                                step=10.0)
                    
                    # Temperature input
                    temp = st.number_input("Average Temperature (°C)", 
                                        min_value=-20.0, 
                                        max_value=40.0, 
                                        value=20.0,
                                        step=0.1)
                    
                    # Crop selection
                    item = st.selectbox("Crop Type", items)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Submit button
                submit = st.form_submit_button("Predict Yield", use_container_width=True)
            
            if submit:
                # Make prediction
                prediction = predict_yield(model, preprocessor, year, rainfall, pesticides, temp, area, item)
                
                # Display prediction
                st.markdown(f"<div class='prediction-result'>Predicted Yield: {prediction:.2f} hg/ha</div>", unsafe_allow_html=True)
                
                # Add some context about the prediction
                st.markdown(f"""
                <div class='info-text'>
                    <strong>Prediction Context:</strong><br>
                    • This yield prediction is for <strong>{item}</strong> in <strong>{area}</strong> for the year <strong>{year}</strong>.<br>
                    • The predicted value is in hectograms per hectare (hg/ha).<br>
                    • 100 hg/ha = 10 kg/ha = 0.01 tonnes/ha
                </div>
                """, unsafe_allow_html=True)
                
                # Find similar records for comparison
                if df is not None:
                    st.markdown("<h3 style='font-size: 1.3rem; margin: 20px 0 10px 0;'>Similar Historical Records:</h3>", unsafe_allow_html=True)
                    filtered_df = df[(df['Item'] == item) & (df['Area'] == area)]
                    if not filtered_df.empty:
                        # Style the dataframe to make it more compact
                        st.dataframe(
                            filtered_df[['Year', 'average_rain_fall_mm_per_year', 
                                         'pesticides_tonnes', 'avg_temp', 
                                         'hg/ha_yield']].sort_values('Year', ascending=False),
                            height=250
                        )
                    else:
                        st.write("No historical records found for this combination.")
        else:
            st.warning("Model not loaded. Please check if model files (dtrt.pkl and preprocessor.pkl) exist.")
    
    with tab2:
        if df is not None:
            st.markdown("<h2 class='yield-subheader'>Data Insights</h2>", unsafe_allow_html=True)
            
            # Show dataset statistics in a more compact layout
            st.markdown("<h3 style='font-size: 1.3rem; margin: 20px 0 10px 0;'>Dataset Overview</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:,}</div>
                    <div class="metric-label">Total Records</div>
                </div>
                """.format(df.shape[0]), unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:,}</div>
                    <div class="metric-label">Countries</div>
                </div>
                """.format(len(df['Area'].unique())), unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:,}</div>
                    <div class="metric-label">Crops</div>
                </div>
                """.format(len(df['Item'].unique())), unsafe_allow_html=True)
            
            # Data visualizations
            st.markdown("<h3 style='font-size: 1.3rem; margin: 30px 0 15px 0;'>Visualizations</h3>", unsafe_allow_html=True)
            
            # Select visualization type
            viz_type = st.selectbox(
                "Select Visualization",
                ["Yield by Country", "Yield by Crop", "Correlation Matrix", "Yield Over Time"]
            )
            
            # Container for visualizations
            st.markdown("<div class='viz-container'>", unsafe_allow_html=True)
            
            if viz_type == "Yield by Country":
                # Create a figure for country yields with improved styling
                plt.style.use('ggplot')
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Limit to top countries for better visibility
                top_n = st.slider("Number of Top Countries to Show", 5, 30, 15)
                
                country_yields = df.groupby('Area')['hg/ha_yield'].mean().sort_values(ascending=False).head(top_n)
                
                # Use a nicer color palette
                bar_colors = sns.color_palette("YlGn", len(country_yields))
                bars = sns.barplot(y=country_yields.index, x=country_yields.values, ax=ax, palette=bar_colors)
                
                # Formatting
                ax.set_title(f"Average Crop Yield by Country (Top {top_n})", fontsize=14, fontweight='bold')
                ax.set_xlabel("Average Yield (hg/ha)", fontsize=12)
                ax.set_ylabel("Country", fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                
                # Add values to the bars
                for i, v in enumerate(country_yields.values):
                    ax.text(v + 100, i, f"{v:.0f}", va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Yield by Crop":
                # Create a figure for crop yields with improved styling
                plt.style.use('ggplot')
                fig, ax = plt.subplots(figsize=(10, 8))
                
                crop_yields = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False)
                
                # Use a nice color palette
                bar_colors = sns.color_palette("YlGn", len(crop_yields))
                bars = sns.barplot(y=crop_yields.index, x=crop_yields.values, ax=ax, palette=bar_colors)
                
                # Formatting
                ax.set_title("Average Yield by Crop Type", fontsize=14, fontweight='bold')
                ax.set_xlabel("Average Yield (hg/ha)", fontsize=12)
                ax.set_ylabel("Crop", fontsize=12)
                ax.tick_params(axis='y', labelsize=9)
                
                # Add values to the bars
                for i, v in enumerate(crop_yields.values):
                    ax.text(v + 500, i, f"{v:.0f}", va='center', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Correlation Matrix":
                # Create correlation matrix with improved styling
                plt.style.use('ggplot')
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Use a better colormap for correlation
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
                
                # Generate heatmap with better formatting
                sns.heatmap(
                    numeric_df.corr(), 
                    annot=True, 
                    cmap=cmap,
                    mask=mask,
                    vmax=1.0, 
                    vmin=-1.0,
                    fmt='.2f',
                    linewidths=0.5,
                    ax=ax,
                    annot_kws={"size": 8}
                )
                
                ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
            elif viz_type == "Yield Over Time":
                # Time series of yield with improved styling
                plt.style.use('ggplot')
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Select specific country and crop
                col1, col2 = st.columns(2)
                with col1:
                    selected_area = st.selectbox("Select Country", sorted(df['Area'].unique()), key="area_ts")
                with col2:
                    selected_item = st.selectbox("Select Crop", sorted(df['Item'].unique()), key="item_ts")
                
                filtered_df = df[(df['Area'] == selected_area) & (df['Item'] == selected_item)]
                
                if not filtered_df.empty:
                    yearly_data = filtered_df.groupby('Year')['hg/ha_yield'].mean().reset_index()
                    
                    # Plot the time series with better styling
                    sns.lineplot(
                        x='Year', 
                        y='hg/ha_yield', 
                        data=yearly_data, 
                        ax=ax, 
                        marker='o',
                        markersize=6,
                        linewidth=2,
                        color='#43A047'
                    )
                    
                    # Add trend line
                    try:
                        z = np.polyfit(yearly_data['Year'], yearly_data['hg/ha_yield'], 1)
                        p = np.poly1d(z)
                        ax.plot(yearly_data['Year'], p(yearly_data['Year']), 
                                linestyle='--', color='#E53935', linewidth=1.5,
                                label=f"Trend (Slope: {z[0]:.2f})")
                        ax.legend()
                    except:
                        pass
                    
                    # Formatting
                    ax.set_title(f"{selected_item} Yield in {selected_area} Over Time", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Year", fontsize=12)
                    ax.set_ylabel("Yield (hg/ha)", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No data available for this combination.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Raw data explorer
            st.markdown("<h3 style='font-size: 1.3rem; margin: 30px 0 10px 0;'>Raw Data Explorer</h3>", unsafe_allow_html=True)
            if st.checkbox("Show Raw Data"):
                st.dataframe(df, height=300)