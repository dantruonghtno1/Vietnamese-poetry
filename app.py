from inference import infer
from flask import Flask, request, render_template, jsonify
import io 
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/', methods = ['GET', 'POST'])
def generation():
    if request.method == "POST":
        text = request.form["input"]
        model = "save_model"
        if request.form['submit_button'] == "Thơ 5 chữ":
            thetho = request.form["submit_button"]
            model = "model_5chu"
        if request.form['submit_button'] == "Thơ 7 chữ":
            thetho = request.form["submit_button"]
            model = "model_7chu"
        if request.form['submit_button'] == "Thơ 8 chữ":
            thetho = request.form["submit_button"]
            model = "model_8chu"
        if request.form['submit_button'] == "Thơ lục bát":
            thetho = request.form["submit_button"]
            model = "model_lucbat"
        
        result = infer(text, model_path= model)
        result = result.strip()
        length = len(result)
        check = 0
        endline = 0
        for i in range(length):
            if result[i] == "\n":
                check+=1
                if check == 8:
                    endline = int(i)
        if endline != 0:
            tho = result[3:endline]
        else:
            tho = result[3:None]

        return render_template('index.html', 
                                text=text, 
                                thetho=thetho,
                                tho=tho)

if __name__ == "__main__":
    app.run(debug=False, host = "0.0.0.0", port = "8000")