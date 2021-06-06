from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
svm=pickle.load(open('churn.pkl','rb'))

@app.route('/')
def home():
    return render_template("churn.html")


@app.route('/predict',methods=['post'])
def predict():
    Creditscore=int(request.form['CREDITSCORE'])
    Geography=int(request.form['GEOGRAPHY'])
    Gender=int(request.form['GENDER'])
    Age=int(request.form['AGE'])
    Tenure=int(request.form['TENURE'])
    Balance=int(request.form['BALANCE'])
    Numofproducts=int(request.form['NUMOFPRODUCTS'])
    Hascard=int(request.form['HASCRCARD'])
    Isactivemember=int(request.form['ISACTIVEMEMBER'])
    Estimatedsalary=int(request.form['ESTIMATEDSALARY'])
    
    a=np.array([[Creditscore,Geography,Gender,Age,Tenure,Balance,Numofproducts,Hascard,Isactivemember,Estimatedsalary]])
    print(a)

    result=svm.predict(a)

    return render_template('churn.html',x=result)

if __name__ == '__main__':
    app.run()