from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained Decision Tree model (Gini index)
# Keep DTModel.pkl in the same folder as app.py
with open('DTModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature order must exactly match the training data
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)

        # Validate JSON input
        if not isinstance(data, dict):
            return jsonify({
                'error': 'Invalid JSON input.'
            }), 400

        # Check for missing values
        missing = [
            feature for feature in FEATURE_ORDER
            if feature not in data
            or data[feature] is None
            or data[feature] == ''
        ]

        if missing:
            return jsonify({
                'error': f'Missing values for: {", ".join(missing)}'
            }), 400

        # Convert input values to numbers in the correct order
        features = [
            float(data[feature])
            for feature in FEATURE_ORDER
        ]

        # Use a DataFrame with feature names to avoid the warning
        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_ORDER
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Calculate predicted class probability
        probabilities = model.predict_proba(input_data)[0]
        confidence = round(
            float(max(probabilities)) * 100, 2
        )

        # Prepare response
        result = {
            'prediction': int(prediction),
            'label': (
                'Heart Disease Detected'
                if prediction == 1
                else 'No Heart Disease Detected'
            ),
            'confidence': confidence
        }

        return jsonify(result)

    except (ValueError, TypeError):
        return jsonify({
            'error': 'All fields must be valid numbers.'
        }), 400

    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({
            'error': 'Prediction failed. Check the server logs.'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)