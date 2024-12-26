from flask import Flask, request, jsonify, render_template

app = Flask(__name__)  # creating an object of Flask class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data using request.form
    form_data = {
        "Number of Dependents": request.form.get("no_of_dependents"),
        "Education": request.form.get("education"),
        "Self Employed": request.form.get("self_employed"),
        "Annual Income": request.form.get("income_annum"),
        "Loan Amount": request.form.get("loan_amount"),
        "Loan Term": request.form.get("loan_term"),
        "CIBIL Score": request.form.get("cibil_score"),
        "Residential Assets Value": request.form.get("residential_assets_value"),
        "Commercial Assets Value": request.form.get("commercial_assets_value"),
        "Luxury Assets Value": request.form.get("luxury_assets_value"),
        "Bank Asset Value": request.form.get("bank_asset_value"),
    }

    # Display the collected data
    return render_template('result.html', form_data=form_data)





if __name__ == '__main__':
    app.run(debug=True)



