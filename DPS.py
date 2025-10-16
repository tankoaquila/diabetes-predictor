# Single-File Flask App for Diabetes Prediction with Login, History, Clear, and Chart (Blue Diabetes Theme)
# Run this in your Python environment: python testModel.py
# Access at http://localhost:5000 in your browser (ALWAYS starts at Login)
# Default Login: Any username with password '1234'
# Ensure 'diabetes_rf_model.pkl' is in the same folder

from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime
import webbrowser

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Required for sessions

# Load the trained model
model = joblib.load('DiabetesPredictionModel.pkl')

# Imputer
features_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(strategy='median')
dummy_df = pd.DataFrame(np.zeros((1, len(features_to_impute))), columns=features_to_impute)
imputer.fit(dummy_df)

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Overall dataset stats (hardcoded from Pima dataset: 500 no diabetes, 268 yes)
OVERALL_STATS = {'no_diabetes': 500, 'diabetes': 268}

# Login Template (Blue Diabetes Theme)
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Prediction System - Login</title>
    <style>
        body { 
            margin: 0; 
            padding: 0; 
            font-family: Arial, sans-serif; 
            background: linear-gradient(to right, #2196F3, #1976D2); 
            display: flex; 
            height: 100vh; 
            align-items: center; 
            justify-content: center; 
            position: relative; 
            overflow: hidden;
        }
        body::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(255,255,255,0.05) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        .login-container {
            display: flex;
            width: 100%;
            max-width: 1200px;
            height: 80vh;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .left-side {
            flex: 1;
            background: linear-gradient(to bottom, #2196F3, #0D47A1);
            border-radius: 20px 0 0 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .left-side::before {
            content: '🩸';  /* Blood drop emoji as placeholder for cells */
            font-size: 200px;
            opacity: 0.3;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .right-side {
            flex: 1;
            background: white;
            border-radius: 0 20px 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }
        .login-form {
            width: 100%;
            max-width: 300px;
        }
        .login-title {
            text-align: center;
            color: #2196F3;
            font-size: 24px;
            margin-bottom: 30px;
            font-weight: bold;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: bold;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="left-side"></div>
        <div class="right-side">
            <form action="/login" method="POST" class="login-form">
                <h2 class="login-title">Diabetes Risk Prediction System</h2>
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit">Login</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# Prediction Template with Sidebar History and Chart (Blue Theme)
PREDICTION_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Prediction</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2196F3;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --light-bg: #f8f9fa;
            --white: #ffffff;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        body { 
            font-family: 'Roboto', sans-serif; 
            background: linear-gradient(135deg, #2196F3 0%, #0D47A1 100%); 
            min-height: 100vh;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        .form-container {
            background: var(--white);
            padding: 40px;
            border-radius: 15px;
            box-shadow: var(--shadow);
        }
        .sidebar {
            background: var(--white);
            padding: 20px;
            border-radius: 15px;
            box-shadow: var(--shadow);
            height: fit-content;
        }
        h1 { 
            color: var(--primary-color); 
            text-align: center; 
            margin-bottom: 10px;
            font-weight: 700;
            font-size: 2em;
        }
        p { 
            text-align: center; 
            color: var(--secondary-color);
            margin-bottom: 30px;
            font-weight: 300;
        }
        form { 
            display: grid; 
            gap: 15px; 
            background: var(--light-bg); 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }
        label { 
            font-weight: 400; 
            color: #555;
            display: block;
            margin-bottom: 5px;
        }
        input { 
            padding: 12px; 
            font-size: 16px; 
            border: 2px solid #e9ecef; 
            border-radius: 8px; 
            transition: border-color 0.3s, box-shadow 0.3s;
            width: 100%;
            box-sizing: border-box;
        }
        input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }
        .button-group {
            display: flex;
            gap: 10px;
        }
        button { 
            background: var(--primary-color); 
            color: var(--white); 
            border: none; 
            cursor: pointer; 
            padding: 12px 24px; 
            font-size: 16px; 
            border-radius: 8px; 
            transition: background 0.3s, transform 0.2s;
            font-weight: 500;
        }
        button:hover { 
            background: #1976D2; 
            transform: translateY(-2px);
        }
        button[type="reset"] {
            background: var(--secondary-color);
        }
        button[type="reset"]:hover {
            background: #545b62;
        }
        #result { 
            margin-top: 30px; 
            padding: 20px; 
            border-radius: 10px; 
            display: none; 
            animation: fadeIn 0.5s ease-in;
            text-align: center;
            font-weight: 500;
        }
        #result.success {
            background: #d4edda; 
            border: 2px solid var(--success-color);
            color: var(--success-color);
        }
        #result.high-risk { 
            background: #f8d7da; 
            border: 2px solid var(--danger-color);
            color: var(--danger-color);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .logout {
            text-align: right;
            margin-bottom: 20px;
        }
        .logout a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
        }
        .logout a:hover {
            text-decoration: underline;
        }
        .history-section {
            margin-bottom: 20px;
        }
        .history-title {
            color: var(--primary-color);
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .clear-btn {
            background: #dc3545;
            width: 100%;
            margin-bottom: 15px;
        }
        .clear-btn:hover {
            background: #c82333;
        }
        .history-list {
            list-style: none;
            padding: 0;
            max-height: 300px;
            overflow-y: auto;
        }
        .history-item {
            background: var(--light-bg);
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .history-item.high { color: var(--danger-color); }
        .history-item.low { color: var(--success-color); }
        #chart-container {
            position: relative;
            height: 200px;
            margin-top: 20px;
        }
        .no-history {
            text-align: center;
            color: var(--secondary-color);
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="form-container">
            <div class="logout"><a href="/logout">Logout</a></div>
            <h1>💉 Diabetes Risk Predictor</h1>
            <p>Enter your health details below to get an instant prediction using our AI-powered model.</p>
            <form id="predictForm">
                <label for="pregnancies">Number of Pregnancies:</label>
                <input type="number" id="pregnancies" name="Pregnancies" required min="0">
                
                <label for="glucose">Glucose Level (mg/dL):</label>
                <input type="number" id="glucose" name="Glucose" required min="0">
                
                <label for="bp">Blood Pressure (mm Hg):</label>
                <input type="number" id="bp" name="BloodPressure" required min="0">
                
                <label for="skin">Skin Thickness (mm):</label>
                <input type="number" id="skin" name="SkinThickness" required min="0">
                
                <label for="insulin">Insulin Level (mu U/ml):</label>
                <input type="number" id="insulin" name="Insulin" required min="0">
                
                <label for="bmi">BMI (kg/m²):</label>
                <input type="number" id="bmi" step="0.1" name="BMI" required min="0">
                
                <label for="pedigree">Diabetes Pedigree Function:</label>
                <input type="number" id="pedigree" step="0.001" name="DiabetesPedigreeFunction" required min="0">
                
                <label for="age">Age (years):</label>
                <input type="number" id="age" name="Age" required min="0">
                
                <div class="button-group">
                    <button type="submit">🔮 Predict Risk</button>
                    <button type="reset">🔄 Reset</button>
                </div>
            </form>
            <div id="result"></div>
        </div>
        <div class="sidebar">
            <div class="history-section">
                <h3 class="history-title">📋 Prediction History</h3>
                <button class="clear-btn" onclick="clearHistory()">🗑️ Clear History</button>
                <ul id="historyList" class="history-list"></ul>
                <div id="noHistory" class="no-history">No predictions yet. Make one to see history!</div>
            </div>
            <div class="chart-section">
                <h3 class="history-title">📊 Diabetes Risk Distribution</h3>
                <div id="chart-container">
                    <canvas id="riskChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let history = [];
        let chart;

        // Initialize chart with overall stats
        function initChart() {
            const ctx = document.getElementById('riskChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['No Diabetes', 'Diabetes'],
                    datasets: [{
                        data: [65, 35],  // Overall %
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' },
                        title: { display: true, text: 'Overall Dataset (65% No / 35% Yes)' }
                    }
                }
            });
        }

        // Update chart from history
        function updateChart() {
            const noDiabetes = history.filter(h => h.outcome === 0).length;
            const diabetes = history.filter(h => h.outcome === 1).length;
            const total = history.length;
            if (total === 0) {
                chart.data.datasets[0].data = [65, 35];
                chart.options.plugins.title.text = 'Overall Dataset (65% No / 35% Yes)';
            } else {
                const noPct = (noDiabetes / total * 100).toFixed(1);
                const yesPct = (diabetes / total * 100).toFixed(1);
                chart.data.datasets[0].data = [noPct, yesPct];
                chart.options.plugins.title.text = `Your History (${noPct}% No / ${yesPct}% Yes)`;
            }
            chart.update();
        }

        // Update history list
        function updateHistoryList() {
            const list = document.getElementById('historyList');
            const noHistoryDiv = document.getElementById('noHistory');
            list.innerHTML = '';
            if (history.length === 0) {
                noHistoryDiv.style.display = 'block';
                return;
            }
            noHistoryDiv.style.display = 'none';
            history.forEach((item, index) => {
                const li = document.createElement('li');
                li.className = `history-item ${item.outcome === 1 ? 'high' : 'low'}`;
                li.innerHTML = `<strong>${new Date(item.timestamp).toLocaleString()}</strong><br>
                                Outcome: ${item.outcome === 1 ? 'Diabetes Positive' : 'No Diabetes'}<br>
                                Probability: ${item.probability}`;
                list.appendChild(li);
            });
        }

        // Clear history
        function clearHistory() {
            fetch('/clear_history', { method: 'POST' })
                .then(response => response.json())
                .then(() => {
                    history = [];
                    updateHistoryList();
                    updateChart();
                });
        }

        // Load initial history
        fetch('/history')
            .then(response => response.json())
            .then(data => {
                history = data;
                updateHistoryList();
                updateChart();
            });

        // Form submit
        const form = document.getElementById('predictForm');
        const resultDiv = document.getElementById('result');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const result = await response.json();
                resultDiv.className = result.outcome === 1 ? 'high-risk' : 'success';
                resultDiv.innerHTML = `
                    <h2>Prediction: ${result.risk} Risk</h2>
                    <p>Outcome: ${result.outcome ? 'Diabetes Positive' : 'No Diabetes'}</p>
                    <p>Probability of Diabetes: ${result.probability}</p>
                `;
                resultDiv.style.display = 'block';

                // Add to history
                result.timestamp = Date.now();
                history.unshift(result);
                updateHistoryList();
                updateChart();
            } catch (error) {
                resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                resultDiv.style.display = 'block';
            }
        });

        form.addEventListener('reset', (e) => {
            resultDiv.style.display = 'none';
        });

        // Init
        initChart();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    # ALWAYS start with login page
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('home'))
    return render_template_string(PREDICTION_TEMPLATE)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if password == '1234':
        session['logged_in'] = True
        if 'history' not in session:
            session['history'] = []
        return redirect(url_for('dashboard'))
    else:
        return 'Invalid password! <a href="/">Try again</a>', 401

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('history', None)
    return redirect(url_for('home'))

@app.route('/history')
def get_history():
    if 'logged_in' not in session:
        return jsonify([])
    return jsonify(session['history'])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'logged_in' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    session['history'] = []
    return jsonify({'status': 'cleared'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('home'))
    try:
        data = request.form.to_dict()
        input_data = np.array([[float(data.get(name, 0)) for name in feature_names]])
        input_df = pd.DataFrame(input_data, columns=feature_names)
        input_df[features_to_impute] = imputer.transform(input_df[features_to_impute])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
        result = {
            'outcome': int(prediction),
            'risk': 'High' if prediction == 1 else 'Low',
            'probability': f"{probability:.2%}" if probability is not None else 'N/A'
        }
        # Append to history
        result['timestamp'] = datetime.now().isoformat()
        session['history'].insert(0, result)  # Prepend for newest first
        session.modified = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

