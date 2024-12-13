import streamlit as st
from PIL import Image
from backend import predict_class  # Import the prediction function from backend.py

st.title('Water Footprint Calculator')

# Allow users to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Add a button to make prediction
    if st.button('Predict'):
        with st.spinner('Classifying...'):
            # Make prediction
            predicted_class ,info= predict_class(img)
            with st.success(''):
                st.markdown(f"**Prediction:** {predicted_class} <br> **Water Footprint Info:** {info}", unsafe_allow_html=True)



# import streamlit as st
# import requests
# from PIL import Image
# import io

# # Define the URL of your Flask API
# API_URL = 'http://localhost:5000/predict'

# st.title("Vegetable Water Footprint Predictor")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image of a vegetable...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the image
#     st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
#     # Convert the uploaded image to a format compatible with the API
#     image = Image.open(uploaded_file)
#     buffer = io.BytesIO()
#     image.save(buffer, format="PNG")
#     buffer.seek(0)
    
#     # Make API request
#     response = requests.post(API_URL, files={"file": buffer})
#     data = response.json()
    
#     # Display the result
#     if 'product_name' in data:
#         st.write(f"Product Name: {data['product_name']}")
#         st.write(f"Description: {data['description']}")
#     else:
#         st.write("Error: Unable to get prediction.")
