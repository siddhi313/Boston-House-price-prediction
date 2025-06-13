from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('house_price_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

    fields = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    form_data = {}
    if request.method == 'POST':
        form_data = {f: request.form.get(f, '') for f in fields}
        features = [float(form_data[f] or 0) for f in fields]
        features_array = np.array([features])
        prediction = round(model.predict(features_array)[0], 2)

    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)


