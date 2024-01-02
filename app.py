import numpy as np 
import pickle
import streamlit as st 
with open('Model.pkl','rb') as f:
    model=pickle.load(f)
with open('Scaler.pkl','rb') as f:
    scaler=pickle.load(f)

location_mapping={0: 'AGIRIPALLI',
 1: 'Ajit Singh Nagar',
 2: 'Andhra Prabha Colony Road',
 3: 'Ashok Nagar',
 4: 'Auto Nagar',
 5: 'Ayyappa Nagar',
 6: 'Bandar Road',
 7: 'Benz Circle',
 8: 'Bharathi Nagar',
 9: 'Bhavanipuram',
 10: 'Chennai Vijayawada Highway',
 11: 'Currency Nagar',
 12: 'Devi Nagar',
 13: 'Edupugallu',
 14: 'Enikepadu',
 15: 'G Konduru',
 16: 'Gandhi Nagar',
 17: 'Gannavaram',
 18: 'Gollapudi',
 19: 'Gollapudi1',
 20: 'Gosala',
 21: 'Governor Peta',
 22: 'Gudavalli',
 23: 'Gunadala',
 24: 'Guntupalli',
 25: 'Guru Nanak Colony',
 26: 'Ibrahimpatnam',
 27: 'Jaggayyapet',
 28: 'Kanchikacherla',
 29: 'Kandrika',
 30: 'Kanigiri Gurunadham Street',
 31: 'Kankipadu',
 32: 'Kanuru',
 33: 'Kesarapalle',
 34: 'Kesarapalli',
 35: 'LIC Colony',
 36: 'Labbipet',
 37: 'Madhuranagar',
 38: 'Mangalagiri',
 39: 'Milk Factory Road',
 40: 'Moghalrajpuram',
 41: 'Mylavaram',
 42: 'MylavaramKuntamukkalaVellaturuVijayawada Road',
 43: 'Nandigama',
 44: 'Nidamanuru',
 45: 'Nunna',
 46: 'Nuzividu',
 47: 'Nuzvid Road',
 48: 'Nuzvid To Vijayawada Road',
 49: 'PNT Colony',
 50: 'Pamarru',
 51: 'Patamata',
 52: 'Payakapuram',
 53: 'Pedapulipaka Tadigadapa Road',
 54: 'Penamaluru',
 55: 'Poranki',
 56: 'Punadipadu',
 57: 'Rajiv Bhargav Colony',
 58: 'Rama Krishna Puram',
 59: 'Ramalingeswara Nagar',
 60: 'Ramavarapadu',
 61: 'Ramavarapadu Ring',
 62: 'SURAMPALLI',
 63: 'Satyanarayanapuram Main Road',
 64: 'Satyaranayana Puram',
 65: 'Sri Ramachandra Nagar',
 66: 'Srinivasa Nagar Bank Colony',
 67: 'Subba Rao Colony 2nd Cross Road',
 68: 'Tadepalligudem',
 69: 'Tadigadapa',
 70: 'Tadigadapa Donka Road',
 71: 'Tarapet',
 72: 'Telaprolu',
 73: 'Tulasi Nagar',
 74: 'Vaddeswaram',
 75: 'Vidhyadharpuram',
 76: 'Vijayawada Airport Road',
 77: 'Vijayawada Guntur Highway',
 78: 'Vijayawada Nuzvidu Road',
 79: 'Vijayawada Road',
 80: 'Vuyyuru',
 81: 'chinnakakani',
 82: 'currency nagar',
 83: 'krishnalanka',
 84: 'kunchanapalli',
 85: 'ramavarappadu'}

location_mapping1={0: 'New', 1: 'Ready to move', 2: 'Resale', 3: 'Under Construction'}

location_mapping2={0: 'East',
 1: 'None',
 2: 'North',
 3: 'NorthEast',
 4: 'NorthWest',
 5: 'South',
 6: 'SouthEast',
 7: 'SouthWest',
 8: 'West'}

location_mapping3={0: 'Apartment',
1: 'Independent Floor',
 2: 'Independent House',
 3: 'Residential Plot',
 4: 'Studio Apartment',
 5: 'Villa'}
import numpy as np
import pickle
import streamlit as st

# Define global dictionaries for mappings
location_mapping = {
    "Poranki": 8,
    "Kankipadu": 5,
    "Benz Circle": 0,
    "Gannavaram": 2,
    "Rajarajeswari Peta": 9,
    "Gunadala": 4,
    "Gollapudi": 3,
    "Enikepadu": 1,
    "Vidhyadharpuram": 10,
    "Penamaluru": 7,
    "Payakapuram": 6
}

status_mapping = {
    "Resale": 2,
    "Under Construction": 3,
    "Ready to move": 1,
    "New": 0
}

direction_mapping = {
    "Not Mentioned": 0,
    "East": 1,
    "West": 3,
    "NorthEast": 2
}

property_type_mapping = {
    "Apartment": 0,
    "Independent Floor": 1,
    "Independent House": 2,
    "Residential Plot": 3
}

with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(bed, bath, loc, size, status, face, Type):
    selected_location_numeric = location_mapping[loc]
    selected_status_numeric = status_mapping[status]
    selected_direction_numeric = direction_mapping[face]
    selected_property_type_numeric = property_type_mapping[Type]

    input_data = np.array([[bed, bath, selected_location_numeric, size, selected_status_numeric, selected_direction_numeric, selected_property_type_numeric]])

    input_df = scaler.transform(input_data)

    return model.predict(input_df)[0]

if __name__ == '__main__':
    st.header('House Price Prediction')

    # Create a column layout to add the image alongside the prediction
    col1, col2 = st.columns([2, 1])

    bed = col1.slider('No of Bedrooms', max_value=10, min_value=1, value=2)
    bath = col1.slider('No of Bathrooms', max_value=7, min_value=1, value=2)
    loc = col1.selectbox("Select a Location", list(location_mapping.keys()))
    size = col1.number_input('Enter the Sq Feet', max_value=10000, min_value=100, value=1000, step=500)
    status = col1.selectbox("Select a Status", list(status_mapping.keys()))
    face = col1.selectbox("Select a Direction", list(direction_mapping.keys()))
    Type = col1.selectbox("Select a Property Type", list(property_type_mapping.keys()))

    result = predict(bed, bath, loc, size, status, face, Type)

    # Add an image to the second column (you need to specify the image URL)
    col2.image('https://img.freepik.com/free-photo/blue-house-with-blue-roof-sky-background_1340-25953.jpg', use_column_width=True)
    
    # Display the predicted value in the first column
    col2.write(f"The predicted value is: {result} Lakhs")