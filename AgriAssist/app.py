from flask import Flask, request, render_template
import numpy as np
from flask import Flask, request, render_template
import numpy as np
import joblib  
from flask import Flask, request, render_template

# Load model and preprocessing tools
model = joblib.load('model/model.pkl')
sc = joblib.load('model/standscaler.pkl')
le = joblib.load('model/labelencoder.pkl')

# Load model and preprocessing objects once when app starts
# Load saved model and preprocessing tools (load only once)
model_fertilizer = joblib.load('model/new-fertilizer/fertilizer_model.pkl')
scaler = joblib.load('model/new-fertilizer/fertilizer_scaler.pkl')
le_soil = joblib.load('model/new-fertilizer/soil_label_encoder.pkl')
le_crop = joblib.load('model/new-fertilizer/crop_label_encoder.pkl')
le_fert = joblib.load('model/new-fertilizer/fert_label_encoder.pkl')

# Creating Flask app
app = Flask(__name__)

# Crop mapping dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/predict", methods=['POST'])
def predict():
    feature_list = [
        float(request.form['Nitrogen']),
        float(request.form['Phosporus']), 
        float(request.form['Potassium']),
        float(request.form['Temperature']),
        float(request.form['Humidity']),
        float(request.form['Ph']),
        float(request.form['Rainfall'])
    ]

    single_pred = np.array(feature_list).reshape(1, -1)
    final_features = sc.transform(single_pred)
    prediction = model.predict(final_features)

    pred_label = prediction[0]
    crop = le.inverse_transform([pred_label])[0]
    result = f"{crop.upper()} is the best crop to be cultivated in the given conditions."
    # crop_image = f"{pred_label}.jpg"  # Adjust if image naming matches crop_dict keys
    crop_image = f"{crop.lower()}.jpg"
    return render_template('crop.html', result=result, crop_image=crop_image)

@app.route('/fertilizer')
def fertilizer_form():
    soil_types = le_soil.classes_.tolist()
    crop_types = le_crop.classes_.tolist()
    return render_template('fertilizer.html', soil_types=soil_types, crop_types=crop_types)

@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        soil_types = le_soil.classes_.tolist()
        crop_types = le_crop.classes_.tolist()
        # Extract inputs
        temperature = float(data.get('Temperature', 0))
        humidity = float(data.get('Humidity', 0))
        moisture = float(data.get('Moisture', 0))
        nitrogen = float(data.get('Nitrogen', 0))
        potassium = float(data.get('Potassium', 0))
        phosphorous = float(data.get('Phosphorous', 0))
        soil_type = data.get('Soil_Type') or data.get('Soil Type')
        crop_type = data.get('Crop_Type') or data.get('Crop Type')

        # Encode inputs
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        features = np.array([[temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_encoded, crop_encoded]])
        features_scaled = scaler.transform(features)

        pred_encoded = model_fertilizer.predict(features_scaled)[0]

        if pred_encoded >= len(le_fert.classes_):
            result = "Prediction out of known class range."
            return render_template('fertilizer.html', result=result, soil_types=soil_types, crop_types=crop_types)
        else:
            pred_fertilizer = le_fert.classes_[pred_encoded]
            if pred_fertilizer == "10-10-10":
                result = f"Predicted fertilizer: 10-10-10\nN-P-K Ratio: 10-10-10 — General-purpose, balanced fertilizer for overall plant health. BALANCED NPK FERTILIZER"
            elif pred_fertilizer == "14-14-14":
                result = f"Predicted fertilizer: 14-14-14\nN-P-K Ratio: 14-14-14 — Used for flowers, vegetables, and lawns needing equal nutrients. BALANCED NPK FERTILIZER"
            elif pred_fertilizer == "15-15-15":
                result = f"Predicted fertilizer: 15-15-15\nN-P-K Ratio: 15-15-15 — All-purpose fertilizer when all nutrients are deficient. BALANCED NPK FERTILIZER"
            elif pred_fertilizer == "17-17-17":
                result = f"Predicted fertilizer: 17-17-17\nN-P-K Ratio: 17-17-17 — Strong multi-nutrient boost. BALANCED NPK FERTILIZER"
            elif pred_fertilizer == "10-26-26":
                result = f"Predicted fertilizer: 10-26-26\nN-P-K Ratio: 10-26-26 — Boosts root and fruit development. PHOSPHORUS-RICH FERTILIZER"
            elif pred_fertilizer == "14-35-14":
                result = f"Predicted fertilizer: 14-35-14\nN-P-K Ratio: 14-35-14 — Ideal for flowering and fruiting crops. PHOSPHORUS-RICH FERTILIZER"
            elif pred_fertilizer == "TSP":
                result = f"Predicted fertilizer: TSP\nN-P-K Ratio: 0-46-0 — Strong phosphorus boost, especially in early growth stages. PHOSPHORUS-RICH FERTILIZER"
            elif pred_fertilizer == "Superphosphate":
                result = f"Predicted fertilizer: Superphosphate\nN-P-K Ratio: 0-20-0 — Mild phosphorus fertilizer suitable for most soils. PHOSPHORUS-RICH FERTILIZER"
            elif pred_fertilizer == "Urea":
                result = f"Predicted fertilizer: Urea\nN-P-K Ratio: 46-0-0 — Fast nitrogen supply to leafy vegetables and nitrogen-deficient soil. NITROGEN-RICH FERTILIZER"
            elif pred_fertilizer == "DAP":
                result = f"Predicted fertilizer: DAP\nN-P-K Ratio: 18-46-0 — Supplies both nitrogen and phosphorus at planting. NITROGEN & PHOSPHORUS FERTILIZER"
            elif pred_fertilizer == "Potassium chloride":
                result = f"Predicted fertilizer: Potassium chloride\nN-P-K Ratio: 0-0-60 — Boosts potassium in crops like tubers and fruits. POTASSIUM-RICH FERTILIZER"
            elif pred_fertilizer == "Potassium sulfate.":
                result = f"Predicted fertilizer: Potassium sulfate\nN-P-K Ratio: 0-0-50 — Suitable for chloride-sensitive crops; provides sulfur. POTASSIUM-RICH FERTILIZER"
            elif pred_fertilizer == "20-20":
                result = f"Predicted fertilizer: 20-20\nN-P-K Ratio: 20-20-0 or 20-20-20 — High-balanced nutrient demand. BALANCED or P-K-RICH FERTILIZER"
            else:
                result = f"Predicted fertilizer: {pred_fertilizer}\nDescription not found in database."

            # Render result back to HTML template
            return render_template('fertilizer.html', result=result, soil_types=soil_types, crop_types=crop_types)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('fertilizer.html', result=error_message)


if __name__ == '__main__':
    app.run(debug=True)
    