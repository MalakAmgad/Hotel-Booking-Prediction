from flask import Flask, request, render_template
from joblib import load
import pickle  
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
#model = load('model.joblib')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the home route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form and ensure the feature names match the model's expected names
    form_data = {
        'number of adults': int(request.form['number_of_adults']),
        'number of weekend nights': int(request.form['number_of_weekend_nights']),
        'number of week nights': int(request.form['number_of_week_nights']),
        'car parking space': int(request.form['car_parking_space']),
        'room type': int(request.form['room_type']),
        'lead time': int(request.form['lead_time']),
        'average price ': float(request.form['average_price']),
        'special requests': float(request.form['special_requests']),
        'month': int(request.form['month']),
        'day': int(request.form['day']),
        'type of meal_Meal Plan 2': int(request.form['type_of_meal_Meal_Plan_2']),
        'type of meal_Not Selected': int(request.form['type_of_meal_Not_Selected']),
        'market segment type_Offline': int(request.form['market_segment_type_Offline']),
        'market segment type_Online': int(request.form['market_segment_type_Online']),
        'year': int(request.form['year'])
    }

    # Convert the form data to a DataFrame to ensure feature names are present
    input_df = pd.DataFrame([form_data])

    # Make prediction using the loaded model
    prediction = model.predict(input_df)[0]
    print(input_df)
    form_data = request.form
    print(form_data)
    print(prediction)
    # Render the result back to the front-end
    # Pass prediction as an integer to the template
    return render_template('index.html', prediction_text=int(prediction), prediction_available=True)



if __name__ == "__main__":
    app.run(debug=True)
