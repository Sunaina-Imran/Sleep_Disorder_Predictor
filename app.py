from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)

#  Generate Synthetic CSV
def generate_csv():
    random.seed(42)
    rows = []
    for i in range(500):  # 500 rows
        phone_usage_after_10pm = random.randint(0, 180)
        blue_light_hours = round(random.uniform(0, 5), 1)
        sleep_duration = round(random.uniform(3, 9), 1)
        phone_pickups_night = random.randint(0, 100)

        night_scrolling = random.choice(['yes', 'no'])
        wake_up_tired = random.choice(['yes', 'no'])
        attention_span = random.choice(['low', 'medium', 'high'])
        breaks_taken = random.choice(['rare', 'sometimes', 'frequent'])

        # Rule-based target
        if phone_usage_after_10pm <= 20 and sleep_duration >= 7 and breaks_taken in ['frequent','sometimes']:
            sleep_disorder = 0  # Healthy Sleep
        elif phone_usage_after_10pm <= 60 and sleep_duration >= 6:
            sleep_disorder = 1  # Slightly Disturbed Sleep
        elif phone_usage_after_10pm <= 120 and sleep_duration >= 4:
            sleep_disorder = 2  # Mild Sleep Disorder
        else:
            sleep_disorder = 3  # Severe Sleep Disorder

        rows.append([
            phone_usage_after_10pm, blue_light_hours, sleep_duration, phone_pickups_night,
            night_scrolling, wake_up_tired, attention_span, breaks_taken, sleep_disorder
        ])

    columns = [
        'phone_usage_after_10pm', 'blue_light_hours', 'sleep_duration', 'phone_pickups_night',
        'night_scrolling', 'wake_up_tired', 'attention_span', 'breaks_taken', 'sleep_disorder'
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("sleep_data.csv", index=False)
    print("✅ CSV Generated: sleep_data.csv")
    return df

# Train Model
def train_model():
    
    if not os.path.exists("sleep_data.csv"):
        df = generate_csv()
    else:
        df = pd.read_csv("sleep_data.csv")

    X = df.drop("sleep_disorder", axis=1)
    y = df["sleep_disorder"]

    numerical_features = ['phone_usage_after_10pm', 'blue_light_hours', 'sleep_duration', 'phone_pickups_night']
    ordinal_features = ['attention_span', 'breaks_taken']
    ordinal_categories = [['low','medium','high'], ['rare','sometimes','frequent']]
    nominal_features = ['night_scrolling','wake_up_tired']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
            ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "sleep_model.pkl")
    print("Model Trained and Saved: sleep_model.pkl")
    return pipeline

# Step 3: Load Model (train if not exist)
if os.path.exists("sleep_model.pkl"):
    model = joblib.load("sleep_model.pkl")
else:
    model = train_model()

label_map = {
    0: "Healthy Sleep ",
    1: "Slightly Disturbed Sleep ",
    2: "Mild Sleep Disorder ",
    3: "Severe Sleep Disorder "
}

#  Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data = {
            "phone_usage_after_10pm": float(request.form["phone_usage_after_10pm"]),
            "blue_light_hours": float(request.form["blue_light_hours"]),
            "sleep_duration": float(request.form["sleep_duration"]),
            "phone_pickups_night": int(request.form["phone_pickups_night"]),
            "night_scrolling": request.form["night_scrolling"],
            "wake_up_tired": request.form["wake_up_tired"],
            "attention_span": request.form["attention_span"],
            "breaks_taken": request.form["breaks_taken"]
        }

        df_input = pd.DataFrame([data])
        pred = model.predict(df_input)[0]
        prediction = label_map[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)