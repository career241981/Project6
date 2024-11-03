from flask import Flask,render_template,request
import pandas as pd 
import pickle

app = Flask(__name__)

# Load the trained model
import os
model_path = r'C:\Users\zubai\Desktop\Data_Science_Jupyter\project-6(Pharmaceutical Sales)\model-30-09-2024-13-38-45-324551.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as e:
        print(f"Error opening file: {e}")
else:
    print(f"File not found: {model_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        customers = int(request.form.get('Customers'))
        open_status = int(request.form.get('Open'))
        weekday = int(request.form.get('Weekday'))
        dayofweek = int(request.form.get('DayOfWeek'))
        isweekend = int(request.form.get('IsWeekend'))

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[customers, open_status, weekday, dayofweek, isweekend]],
                                  columns=['Customers', 'Open', 'Weekday', 'DayOfWeek', 'IsWeekend'])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True, port=8000)