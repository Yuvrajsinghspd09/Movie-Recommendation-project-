from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Your Flask routes and logic go here

if __name__ == '__main__':
    app.run(debug=True)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])

    # Your recommendation logic goes here

    return render_template('recommendations.html', recommendations=recommendations)
