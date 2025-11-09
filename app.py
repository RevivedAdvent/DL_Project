from flask import Flask, render_template, request, redirect
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "models/hotcold_model.keras"
IMG_SIZE = (224, 224)

app = Flask(__name__)

model = load_model(MODEL_PATH, compile=False)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    f = request.files['file']
    if f.filename == '':
        return redirect(request.url)

    try:
        img = Image.open(f).convert("RGB").resize(IMG_SIZE)
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        label = "hot" if preds[0][0] >= 0.5 else "cold"
        conf = float(preds[0][0] if preds[0][0] >= 0.5 else 1 - preds[0][0])

        return render_template("result.html", label=label, confidence=f"{conf:.2%}")
    except Exception as e:
        return f"Error processing image: {e}"

if __name__ == "__main__":
    app.run(debug=True)
