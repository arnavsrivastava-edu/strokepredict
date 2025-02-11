from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import h2o
import pandas as pd
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
h2o.init()
model = h2o.load_model('/Users/arnavsrivastava/PycharmProjects/SchoolProject/model/StackedEnsemble_AllModels_1_AutoML_1_20250131_222515')
@app.route('/')
def home():
    return render_template('program.html')
@app.route('/predict', methods=['POST'])


def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)
        input_data = h2o.H2OFrame(pd.DataFrame([data]))
        prediction = model.predict(input_data)
        prediction_list = prediction.as_data_frame().values.flatten().tolist()

        return jsonify({'prediction': prediction_list})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=12345, debug=True)
