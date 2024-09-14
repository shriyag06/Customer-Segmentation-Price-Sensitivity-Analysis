import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from flask import Flask,request,jsonify,render_template,redirect,url_for

app = Flask(__name__, template_folder = 'templates')


scaler = pickle.load(open('ss.pickle', 'rb'))
pca = pickle.load(open('pca.pickle','rb'))
kmeans_pca = pickle.load(open('kmeans_pca.pickle','rb'))

WO_pp_model = pickle.load(open('WO_pp_model.pickle','rb'))
CF_pp_model = pickle.load(open('CF_pp_model.pickle','rb'))
Standard_pp_model = pickle.load(open('Standard_pp_model.pickle','rb'))
FO_pp_model = pickle.load(open('FO_pp_model.pickle','rb'))
model_ownBrand2=pickle.load(open('model_ownBrand2.pickle','rb'))
model_ownBrand2_cf = pickle.load(open('model_ownBrand2_cf.pickle','rb'))
model_ownBrand2_fo = pickle.load(open('model_ownBrand2_fo.pickle','rb'))
model_ownBrand2_sd=pickle.load(open('model_ownBrand2_sd.pickle','rb'))

@app.route("/")
def base():
    return render_template('base.html')
    

@app.route('/generic', methods=['GET', 'POST'])
def get_segment():
    if request.method == 'POST':
        age = int(request.form.get('age'))
        income = int(request.form.get('income'))
        gender = request.form.get('gender')
        marital_status = request.form.get('marital_s')
        occupation = request.form.get('occupation')
        education = request.form.get('education')
        settlement_size = request.form.get('settlementsize')

        gender = 0 if gender == 'Male' else 1
        marital_status = 0 if marital_status == 'Single' else 1
        education_dict = {'Other': 0, 'High School': 1, 'University': 2, 'Graduate School': 3}
        education = education_dict[education]
        occupation_dict = {'Unemployed': 0, 'Skilled Employee': 1, 'Management': 2}
        occupation = occupation_dict[occupation]
        settlement_size_dict = {'Small City': 0, 'Mid-Size City': 1, 'Big City': 2}
        settlement_size = settlement_size_dict[settlement_size]

        scaled_data = scaler.transform([[gender, marital_status, age, education, income, occupation, settlement_size]])
        pca_data = pca.transform(scaled_data)
        seg_data = kmeans_pca.predict(pca_data)

        seg_dict = {
            0: 'standard',
            1: 'career focused',
            2: 'well-off',
            3: 'fewer opportunities'
        }

        result = seg_dict[seg_data.item()]

        return render_template('generic.html', results=result)
    else:
        return render_template('generic.html')


@app.route('/well-off_price', methods = ['GET','POST'])
def WO_price():
    if request.method == 'GET':
        return render_template('price.html')

    else:
        pp = request.form.get('pp')
        pp = float(pp)
        price_range = np.arange(0.50, 3.50, 0.01)
        df_price_range = pd.DataFrame(price_range)
        Y_pr = WO_pp_model.predict_proba(df_price_range)
        purchase_pr = Y_pr[:][:, 1]
        pe = WO_pp_model.coef_[:, 0] * price_range * (1 - purchase_pr)
        df_price_elasticities = pd.DataFrame(price_range)
        df_price_elasticities = df_price_elasticities.rename(columns = {0: "Price_Point"})
        df_price_elasticities['Mean_PE'] = pe


        pe_r = df_price_elasticities['Mean_PE'].values[0]
        
        if pe_r < 1:
            price_result = "Inelastic"
        else:
            price_result = "Elastic"
        return render_template('price.html', price_results = price_result )
        
@app.route('/Career_price',methods =['GET','POST']) 
def CF_price():
    if request.method == 'GET':
        return render_template('price.html')
        
    else:
        pp1 = request.form.get('pp1')
        pp1 = float(pp1)
        price_range = np.arange(0.5, 3.5, 0.01)
        df_price_range = pd.DataFrame(price_range)
        Y_pr = CF_pp_model.predict_proba(df_price_range)
        purchase_pr = Y_pr[:][:, 1]
        pe = CF_pp_model.coef_[:, 0] * price_range * (1 - purchase_pr)
        df_price_elasticities = pd.DataFrame(price_range)
        df_price_elasticities = df_price_elasticities.rename(columns = {0: "Price_Point"})
        df_price_elasticities['Mean_PE'] = pe
        tolerance = 1e-6
        df_price_elasticities['is_close'] = df_price_elasticities['Price_Point'].apply(lambda x: np.isclose(x, pp1, atol=tolerance))
        df_price_elasticities =  df_price_elasticities[df_price_elasticities['is_close']]
        pe_r = df_price_elasticities['Mean_PE'].values[0]
        
        if pe_r < 1:
            price_result1 = "Inelastic"
        else:
            price_result1 = "Elastic"
        return render_template('price.html', price_result1 = price_result1)

@app.route('/Standard_price',methods =['GET','POST']) 
def S_price():
    if request.method == 'GET':
        return render_template('price.html')
        
    else:
        pp2 = request.form.get('pp2')
        pp2 = float(pp2)
        price_range = np.arange(0.5, 3.5, 0.01)
        df_price_range = pd.DataFrame(price_range)
        Y_pr = Standard_pp_model.predict_proba(df_price_range)
        purchase_pr = Y_pr[:][:, 1]
        pe = Standard_pp_model.coef_[:, 0] * price_range * (1 - purchase_pr)
        df_price_elasticities = pd.DataFrame(price_range)
        df_price_elasticities = df_price_elasticities.rename(columns = {0: "Price_Point"})
        df_price_elasticities['Mean_PE'] = pe
        tolerance = 1e-6
        df_price_elasticities['is_close'] = df_price_elasticities['Price_Point'].apply(lambda x: np.isclose(x, pp2, atol=tolerance))
        df_price_elasticities =  df_price_elasticities[df_price_elasticities['is_close']]
        pe_r = df_price_elasticities['Mean_PE'].values[0]
        if pe_r < 1:
            price_result2 = "Inelastic"
        else:
            price_result2 = "Elastic"
        return render_template('price.html', price_result2 = price_result2)

@app.route('/Fewer_price',methods =['GET','POST']) 
def FO_price():
    if request.method == 'GET':
        return render_template('price.html')
        
    else:
        pp3 = request.form.get('pp3')
        pp3 = float(pp3)
        price_range = np.arange(0.5, 3.5, 0.01)
        df_price_range = pd.DataFrame(price_range)
        Y_pr = FO_pp_model.predict_proba(df_price_range)
        purchase_pr = Y_pr[:][:, 1]
        pe = FO_pp_model.coef_[:, 0] * price_range * (1 - purchase_pr)
        df_price_elasticities = pd.DataFrame(price_range)
        df_price_elasticities = df_price_elasticities.rename(columns = {0: "Price_Point"})
        df_price_elasticities['Mean_PE'] = pe
        tolerance = 1e-6
        df_price_elasticities['is_close'] = df_price_elasticities['Price_Point'].apply(lambda x: np.isclose(x, pp3, atol=tolerance))
        df_price_elasticities =  df_price_elasticities[df_price_elasticities['is_close']]
        pe_r = df_price_elasticities['Mean_PE'].values[0]
        if pe_r < 1:
            price_result3 = "Inelastic"
        else:
            price_result3 = "Elastic"
        return render_template('price.html', price_result3 = price_result3 )
   

 

   
if __name__ == "__main__":
  app.run(debug=True, host = '0.0.0.0',port =8000)


