import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# Set page configuration
st.set_page_config(page_title="Laptop Recommendation System", layout="wide")

# Load and process data
@st.cache_data
def load_data():
    data = pd.read_csv("processed_flipkart_laptops.csv")
    return data

def process_data(data):
    # Create label encoder
    le = LabelEncoder()
    
    # Encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Product Name':  # Keep product name as is
            data[col] = le.fit_transform(data[col].astype(str))
    
    # Handle missing values in Price (₹)
    data['Price (₹)'] = data['Price (₹)'].fillna(data['Price (₹)'].mean())
    return data

# Load data
data = load_data()
processed_data = process_data(data.copy())

# Main title
st.title("Laptop Recommendation System")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a Page", ["EDA & Visualizations", "Laptop Recommender", "Price Predictor"])

if page == "EDA & Visualizations":
    st.header("Exploratory Data Analysis")
    
    # Distribution of Laptop Prices
    st.subheader("Distribution of Laptop Prices")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='Price (₹)', kde=True, bins=20)
    st.pyplot(fig1)
    
    # Price vs RAM
    st.subheader("Price vs RAM")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='RAM (GB)', y='Price (₹)', hue='Processor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)
    
    # Average Price by Processor
    st.subheader("Average Price by Processor")
    avg_price = data.groupby('Processor')['Price (₹)'].mean().sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    avg_price.plot(kind='barh')
    st.pyplot(fig3)
    
    # Ratings Distribution
    st.subheader("Distribution of Ratings")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='Rating', bins=20)
    st.pyplot(fig4)

elif page == "Laptop Recommender":
    st.header("Laptop Recommendation System")
    
    # Get user inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ram = st.selectbox("Select RAM (GB)", sorted(data['RAM (GB)'].unique()))
        processor_type = st.selectbox("Select Processor Type", sorted(data['Processor'].unique()))
    
    with col2:
        storage = st.selectbox("Select Storage (GB)", sorted(data['Storage (GB)'].unique()))
        max_price = st.number_input("Maximum Price (₹)", min_value=0, max_value=int(data['Price (₹)'].max()), value=50000)
    
    if st.button("Get Recommendations"):
        # Filter based on user preferences
        recommendations = data[
            (data['RAM (GB)'] == ram) &
            (data['Storage (GB)'] == storage) &
            (data['Processor'] == processor_type) &
            (data['Price (₹)'] <= max_price)
        ]
        
        if len(recommendations) > 0:
            st.subheader("Recommended Laptops:")
            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['Product Name']} - ₹{row['Price (₹)']:,.2f}"):
                    st.write(f"RAM: {row['RAM (GB)']}GB")
                    st.write(f"Storage: {row['Storage (GB)']}GB")
                    st.write(f"Processor: {row['Processor']}")
                    st.write(f"Rating: {row['Rating']}")
                    st.write(f"Number of Reviews: {row['Number of Reviews']}")
        else:
            st.warning("No laptops found matching your criteria. Try adjusting your preferences.")

else:  # Price Predictor page
    st.header("Laptop Price Predictor")
    
    # Prepare data for model
    # Handle missing values in features
    X = processed_data[['RAM (GB)', 'Storage (GB)', 'Rating', 'Number of Ratings', 'Processor', 'Operating System']]
    X = X.fillna(0)  # Replace missing values in features with 0 (or another strategy)  

# Handle missing values in the target variable
    y = processed_data['Price (₹)']
    y = y.fillna(y.mean())  # Replace missing values in the target with the mean

    
    # Train model
    @st.cache_resource
    def train_model():
    # Handle missing values in features
        X = processed_data[['RAM (GB)', 'Storage (GB)', 'Rating', 'Number of Ratings', 'Processor', 'Operating System']]
        X = X.fillna(0)  # Replace NaNs with 0 (or use a different strategy)

        # Handle missing values in target
        y = processed_data['Price (₹)']
        y = y.fillna(y.mean())  # Replace NaNs in target with the mean

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    
    model = train_model()
    
    # Get user inputs for prediction
    col1, col2 = st.columns(2)
    
    with col1:
        pred_ram = st.selectbox("RAM (GB)", sorted(data['RAM (GB)'].unique()), key='pred_ram')
        pred_storage = st.selectbox("Storage (GB)", sorted(data['Storage (GB)'].unique()), key='pred_storage')
        pred_rating = st.slider("Expected Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    
    with col2:
        pred_processor = st.selectbox("Processor", sorted(processed_data['Processor'].unique()), key='pred_processor')
        pred_os = st.selectbox("Operating System", sorted(processed_data['Operating System'].unique()), key='pred_os')
        pred_ratings_count = st.number_input("Number of Ratings", min_value=0, value=100)
    
    if st.button("Predict Price"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'RAM (GB)': [pred_ram],
            'Storage (GB)': [pred_storage],
            'Rating': [pred_rating],
            'Number of Ratings': [pred_ratings_count],
            'Processor': [pred_processor],
            'Operating System': [pred_os]
        })
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Display prediction
        st.success(f"Predicted Price: ₹{predicted_price:,.2f}")
        
        # Show feature importance
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feat_imp, x='Importance', y='Feature')
        st.pyplot(fig)

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")