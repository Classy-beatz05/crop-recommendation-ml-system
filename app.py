from flask import Flask, request, render_template
import numpy as np
import pickle

# Load model + scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read values from form (names MUST match your HTML)
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])   # spelling matches name in HTML
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        ec = float(request.form['EC'])

        # Order must match training: N, P, K, temperature, humidity, ph, rainfall, ec
        features = [N, P, K, temp, humidity, ph, rainfall, ec]
        arr = np.array(features).reshape(1, -1)

        # Scale features
        scaled = ms.transform(arr)
        final_features = sc.transform(scaled)

        # ---- Top-3 predictions ----
        probs = model.predict_proba(final_features)[0]
        classes = model.classes_

        sorted_idx = np.argsort(probs)[::-1]
        top3_idx = sorted_idx[:3]

        lines = []
        for i in top3_idx:
            crop_name = str(classes[i]).capitalize()
            crop_prob = round(probs[i] * 100, 2)
            lines.append(f"{crop_name} ({crop_prob}%)")

        result = "Top recommended crops:<br>" + "<br>".join(lines)

    except Exception as e:
        # If anything fails, show error in UI (helps debugging)
        result = f"Error while making prediction: {e}"

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
