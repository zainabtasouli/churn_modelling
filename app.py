from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
with open("/Users/mac/Downloads/app machine learning/random_forest_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Récupérer dynamiquement les noms de colonnes attendues
if hasattr(model, 'feature_names_in_'):
    model_columns = list(model.feature_names_in_)
else:
    raise RuntimeError("Ton modèle ne contient pas `feature_names_in_`. "
                       "Réentraîne-le ou charge un fichier 'model_columns.pkl'.")

print("Le modèle attend", model.n_features_in_, "features :", model_columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.form
        # 1. Lecture des inputs
        credit_score = float(f['CreditScore'])
        age          = float(f['Age'])
        tenure       = float(f['Tenure'])
        balance      = float(f['Balance'])
        products     = float(f['NumOfProducts'])
        has_card     = int(f['HasCrCard'])
        is_active    = int(f['IsActiveMember'])
        salary       = float(f['EstimatedSalary'])

        # Genre
        gender_input = f['Gender'].lower()
        if gender_input not in ('male','female'):
            raise ValueError("Genre invalide.")
        gender = 1 if gender_input=='male' else 0

        # Géographie
        geo = f['Geography']
        if geo not in ('France','Germany','Spain'):
            raise ValueError("Pays invalide.")
        # code brut si besoin
        geo_code = {'France':1, 'Spain':2, 'Germany':3}[geo]
        # dummies
        geo_germany = 1 if geo=='Germany' else 0
        geo_spain   = 1 if geo=='Spain'   else 0

        # 2. Préparer un dict de **toutes** les variables potentielles
        inputs = {
            'CreditScore':         credit_score,
            'Gender':              gender,
            'Age':                 age,
            'Tenure':              tenure,
            'Balance':             balance,
            'NumOfProducts':       products,
            'HasCrCard':           has_card,
            'IsActiveMember':      is_active,
            'EstimatedSalary':     salary,
            # soit colonne brute...
            'Geography':           geo_code,
            # …soit dummies
            'Geography_Germany':   geo_germany,
            'Geography_Spain':     geo_spain
        }

        # 3. Construire la liste de features dans le **même ordre** que model_columns
        feature_list = [ inputs[col] for col in model_columns ]

        # 4. Vérifier la dimension
        features = np.array([feature_list])
        if features.shape[1] != model.n_features_in_:
            raise ValueError(f"Le modèle attend {model.n_features_in_} features, "
                             f"on en a {features.shape[1]}.")

        # 5. Prédiction
        pred = model.predict(features)[0]
        msg  = "QUITTER" if pred==1 else "RESTER"
        return render_template('index.html', prediction=f"Client va {msg}")

    except Exception as e:
        print("Erreur dans /predict :", e)
        return render_template('index.html', prediction=f"Erreur : {e}")

if __name__ == "__main__":
    print("Application lancée sur http://127.0.0.1:5000")
    app.run(debug=True)
