from flask import Flask, request, jsonify, render_template

app = Flask(__name__)  # creating an object of Flask class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return f"Working Go Ahead"





if __name__ == '__main__':
    app.run(debug=True)



