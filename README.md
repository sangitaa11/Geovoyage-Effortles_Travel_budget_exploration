
# Travel Budget Prediction Application

## Overview
This Flask-based web application predicts travel budgets based on user input and suggests destinations within the predicted budget. Users can register, log in, provide travel information, and receive budget predictions.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)


## Prerequisites
- Python 3.x
- SQLite (for the included database)
- pip package manager

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/travel-budget-app.git
   cd travel-budget-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

3. Explore the different features of the application, such as user registration, budget prediction, and destination suggestions.

## Project Structure
- **app.py:** The main Flask application file.
- **templates/:** HTML templates for rendering web pages.
- **trained_model.joblib:** The pre-trained machine learning model for budget prediction.
- **preprocessed_travel.csv:** Preprocessed travel data used by the model.
- **site.db:** SQAlchemy database file for user information.

## Dependencies
- Flask
- Flask-SQLAlchemy
- Flask-Login
- Flask-Bcrypt
- scikit-learn
- pandas
- numpy

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.



