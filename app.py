import os, pickle, traceback
import numpy as np
from flask import Flask, request, render_template, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

print("[DEBUG] BASE_DIR:", BASE_DIR)

def try_load_pickle(name):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print(f"[DEBUG] NOT FOUND: {name}")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[DEBUG] LOADED: {name}")
        return obj
    except Exception as e:
        print(f"[DEBUG] ERROR loading {name}:", e)
        traceback.print_exc()
        return None

model = try_load_pickle("model.pkl")
minmax_scaler = try_load_pickle("minmaxscaler.pkl")
stand_scaler = try_load_pickle("standscaler.pkl")
label_encoder = try_load_pickle("label_encoder.pkl")

# if any of those is None, print clear message
print("Model loaded?:", model is not None)
print("MinMax scaler loaded?:", minmax_scaler is not None)
print("Standard scaler loaded?:", stand_scaler is not None)
print("Label encoder loaded?:", label_encoder is not None)

# fallback DummyModel if model not loaded (keeps app running)
if model is None:
    class DummyModel:
        def predict(self, X):
            return [0] * (len(X) if hasattr(X, "__len__") else 1)
    model = DummyModel()
    print("[DEBUG] Using DummyModel (always returns index 0).")

def parse_float(val, default=None):
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except:
        try:
            return float(str(val).replace(",", ""))
        except:
            return default

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = parse_float(request.form.get("Nitrogen"), None)
        P = parse_float(request.form.get("Phosporus"), None)
        if P is None:
            P = parse_float(request.form.get("Phosphorus"), None)
        K = parse_float(request.form.get("Potassium"), None)
        temp = parse_float(request.form.get("Temperature"), None)
        humidity = parse_float(request.form.get("Humidity"), None)
        ph = parse_float(request.form.get("pH"), None)
        rainfall = parse_float(request.form.get("Rainfall"), None)

        required = {"Nitrogen": N, "Phosphorus": P, "Potassium": K,
                    "Temperature": temp, "Humidity": humidity, "pH": ph, "Rainfall": rainfall}
        missing = [k for k,v in required.items() if v is None]
        if missing:
            return render_template("index.html", error=f"Missing or invalid: {', '.join(missing)}")

        X = np.array([[N,P,K,temp,humidity,ph,rainfall]], dtype=float)

        # apply saved scalers if available
        if minmax_scaler is not None:
            try:
                X = minmax_scaler.transform(X)
            except Exception as e:
                print("[DEBUG] minmax transform failed:", e)
        if stand_scaler is not None:
            try:
                X = stand_scaler.transform(X)
            except Exception as e:
                print("[DEBUG] stand scaler transform failed:", e)

        pred = model.predict(X)
        pred_idx = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
        print("[DEBUG] raw model prediction index:", pred_idx)

        if label_encoder is not None:
            # label_encoder.inverse_transform expects array-like of encoded labels
            try:
                crop = label_encoder.inverse_transform([pred_idx])[0]
            except Exception as e:
                print("[DEBUG] label_encoder inverse_transform failed:", e)
                # fallback to string conversion
                crop = str(pred_idx)
        else:
            # If no encoder, try static mapping (old style)
            crop_map = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }
            # try to use mapping; if model output indices are 0-based, adjust
            crop = crop_map.get(pred_idx, crop_map.get(pred_idx+1, str(pred_idx)))

        result = f"{crop} is the best crop to be cultivated right there"
        return render_template("index.html", result=result, inputs=required)

    except Exception as e:
        tb = traceback.format_exc()
        print("[DEBUG] Exception in predict:", e)
        print(tb)
        return render_template("index.html", error=str(e), tb=tb)

if __name__ == "__main__":
    app.run(debug=True)
