import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return """
    <h1>Student Performance Prediction</h1>

    <form action="/predict" method="post">
        Study Hours:
        <input type="number" name="hours" min="0" max="24"
               step="0.5" required><br><br>

        Attendance (%):
        <input type="number" name="attendance" min="0" max="100"
               step="0.1" required><br><br>

        Previous Marks:
        <input type="number" name="marks" min="0" max="100"
               step="0.1" required><br><br>

        <button type="submit">Predict Marks</button>
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():
    try:
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        marks = float(request.form["marks"])

        if not 0 <= hours <= 24:
            return "Study hours must be between 0 and 24.", 400

        if not 0 <= attendance <= 100:
            return "Attendance must be between 0 and 100.", 400

        if not 0 <= marks <= 100:
            return "Previous marks must be between 0 and 100.", 400

        prediction = model.predict([[hours, attendance, marks]])[0]

        prediction = max(0, min(100, prediction))

        return f"""
        <h2>Predicted Marks: {prediction:.2f}</h2>
        <a href="/">Go Back</a>
        """

    except (ValueError, KeyError):
        return "Please enter valid numeric values.", 400


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
