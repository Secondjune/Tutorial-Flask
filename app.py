from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
# Muat model sekali saja tanpa open()
ml_model = joblib.load("model/hasil_pelatihan_model.pkl")

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    print("Prediksi dimulai")
    if request.method == 'POST':
        try:
            # Ambil nilai dari form
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])
            
            # Buat array dari nilai input
            pred_args = [RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)
            
            # Prediksi menggunakan model
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        
        except ValueError:
            # Jika ada error pada input data
            return "Please check if the values are entered correctly"
        
        # Kirim hasil prediksi ke template
        return render_template('predict.html', prediction=model_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
