# app.py
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bcrypt import Bcrypt
from flask_login import current_user

app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the trained model
model = joblib.load('trained_model.joblib')

# Load the travel data+
travel_data = pd.read_csv('preprocessed_travel.csv')

# Convert categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'travel_preferences', 'purpose', 'accommodation_type', 'transportation_mode']
for column in categorical_columns:
    travel_data[column] = label_encoder.fit_transform(travel_data[column])

# Select relevant features for the budget prediction model
budget_features = ['age', 'duration', 'current_cost_of_living_index', 'destination_cost_of_living_index',
                   'accommodation_cost_per_night', 'transportation_cost', 'daily_meal_expenses',
                   'gender', 'travel_preferences', 'purpose', 'actual_expenses']

X_budget = travel_data[budget_features]

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return render_template('main_options.html')  # Render the template directly
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('Please fill in all the fields.', 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/get_budget')
@login_required
def get_budget():
    return render_template('get_budget.html')


@app.route('/predict_budget', methods=['POST'])
@login_required
def predict_budget():
    user_input = request.form.to_dict()

    # Convert input values to numeric
    numeric_input_data = [float(user_input['age']), float(user_input['duration']),
                          float(user_input['current_cost_of_living_index']),
                          float(user_input['destination_cost_of_living_index']),
                          float(user_input['accommodation_cost_per_night']),
                          float(user_input['transportation_cost']),
                          float(user_input['daily_meal_expenses']),
                          float(user_input['gender']), float(user_input['travel_preferences']),
                          float(user_input['purpose']), float(user_input['actual_expenses'])]

    # Use the trained model to predict the budget
    predicted_budget = model.predict([numeric_input_data])[0]

    return render_template('show_budget.html', predicted_budget=predicted_budget)


@app.route('/see_destination')
@login_required
def see_destination():
    return render_template('see_destination.html')


@app.route('/suggest_destination', methods=['POST'])
@login_required
def suggest_destination():
    user_input = request.form.to_dict()
    user_budget = float(user_input['budget'])

    # Suggest a destination based on the provided budget
    suggested_destination = suggest_destination_based_on_budget(user_budget)

    return render_template('show_destination.html', suggested_destination=suggested_destination)


def suggest_destination_based_on_budget(predicted_budget):
    # Your logic for suggesting a destination based on the budget
    if predicted_budget < 10000:
        budget_range = (0, 10000)
    elif 10000 <= predicted_budget < 20000:
        budget_range = (10000, 20000)
    elif 20000 <= predicted_budget < 30000:
        budget_range = (20000, 30000)
    else:
        budget_range = (30000, np.inf)

    # Filter the dataset based on the budget range
    filtered_data = travel_data[
        (travel_data['actual_expenses'] >= budget_range[0]) &
        (travel_data['actual_expenses'] < budget_range[1])
    ]

    # Randomly select a destination from the filtered data
    selected_entry = filtered_data.sample(1).iloc[0]

    # Get details of the selected entry
    selected_destination = selected_entry['destination']
    selected_actual_expenses = selected_entry['actual_expenses']

    return f'{selected_destination} (Actual Expenses: {selected_actual_expenses})'

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/about_us')
def about_us():
    return render_template('about_us.html')


@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')
    
    # Process the feedback as needed (e.g., store it in a database)

    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('contact_us'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
