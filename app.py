import pickle
import streamlit as st 
import os
import numpy as np 

st.title('Housing Price Prediction')
st.write("This is a simple web app to predict the price of a house based on some feature: area, bedroom, bathroom, stories, mainroad, guestroom, basement,hotwaterheating,..v..v ")

@st.cache_resource
def load_all():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path,'notebook','gradient_boost.pkl')
    encoder_path = os.path.join(base_path,'notebook','ordinal_encoder.pkl')
    scaler_path = os.path.join(base_path,'notebook','norm.pkl')

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    return model, encoder, scaler

model, encoder, scaler = load_all()

#input UI
area = st.number_input("Enter your areas: ", value=500, step = 1, min_value = 100, max_value = 100000)
bedrooms = st.number_input("Enter your bedrooms: ", value=2, step = 1, min_value = 1, max_value = 10)
bathrooms = st.number_input("Enter your bathrooms: ", value=2, step = 1, min_value = 1, max_value = 10)
stories = st.number_input("Enter your stories: ", value = 2, step = 1, min_value = 1, max_value = 20)
parking = st.number_input("Enter your parking: ", value=2, step = 1, min_value = 1, max_value = 20)

mainroad = st.selectbox('Main Road?', ['yes', 'no'])
guestroom = st.selectbox('Guest Room?', ['yes', 'no'])
basement = st.selectbox('Basement?', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating?', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning?', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area?', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

# Dự đoán
if st.button('Predict'):
    # Tạo array input
    input_data = [[
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, hotwaterheating, airconditioning,
        prefarea, furnishingstatus
    ]]

    # Chuyển về numpy để xử lý
    input_data = np.array(input_data, dtype=object)

    # Áp dụng encoder
    input_data[:, 5:] = encoder.transform(input_data[:, 5:])

     # 3. Áp dụng StandardScaler cho toàn bộ
    input_scaled = scaler.transform(input_data.astype(float))

    # 4. Dự đoán
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"🏷️ Predicted House Price: ${predicted_price:,.2f}")

