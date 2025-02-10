from flask import Flask, render_template, request, jsonify
import h2o
app = Flask(__name__)
h2o.init()
model = h2o.load_model('model/StackedEnsemble_AllModels_1_AutoML_1_20250131_222515')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = h2o.H2OFrame(data)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.as_data_frame().values.tolist()})
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

