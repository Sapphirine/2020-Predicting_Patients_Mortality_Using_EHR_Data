import flask
import pickle
import pandas as pd
import numpy as np
# Use pickle to load in the pre-trained model.
with open(f'model/patient_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
data=pd.read_csv('df_final6.csv')

gerd=data.columns[1:5]
data[gerd] = data[gerd].astype('category')
measurements=data.columns[5:25]
data[measurements] = data[measurements].applymap(lambda x: np.nan if x=='none' else x)
data[measurements]=data[measurements].astype('float')
categories=data.columns[26:34]
data[categories]=data[categories].astype('category')
condition_type=data.columns[35]
data[condition_type]=data[condition_type].astype('category')

x = data.drop(['death'],axis=1)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        year = int(flask.request.form['year'])
        drug = float(flask.request.form['drug'])
        condition=int(flask.request.form['condition'])
        procedure=float(flask.request.form['procedure'])
        gender=int(flask.request.form['gender'])
        race=int(flask.request.form['race'])
        ethnicity=int(flask.request.form['ethnicity'])
        walking=float(flask.request.form['walking'])
        diabetic=float(flask.request.form['diabetic'])
        highrisk=float(flask.request.form['highrisk'])
        postoperative=float(flask.request.form['postoperative'])
        familyhistory=float(flask.request.form['postoperative'])
        aftercare=float(flask.request.form['aftercare'])
        historyclinical=float(flask.request.form['historyclinical'])
        pacemaker=float(flask.request.form['pacemaker'])
        conditiontype=int(flask.request.form['conditiontype'])

        templist=[year,gender,ethnicity]+[race]+[np.nan]*20+[drug,walking,diabetic,highrisk,postoperative,familyhistory,aftercare,historyclinical,pacemaker,condition,conditiontype,procedure]
        input_variables=pd.DataFrame([templist],columns=x.columns)
        tempdataframe=pd.concat([x,input_variables])
        tempdataframe[tempdataframe.columns[1:4]]=tempdataframe[tempdataframe.columns[1:4]].astype('category')
        tempdataframe[tempdataframe.columns[25:33]]=tempdataframe[tempdataframe.columns[25:33]].astype('category')
        tempdataframe[tempdataframe.columns[34]]=tempdataframe[tempdataframe.columns[34]].astype('category')
        finaldataframe=pd.get_dummies(tempdataframe)
        finaldataframe=finaldataframe.fillna(finaldataframe.median())
        test=finaldataframe.tail(1)

        prediction = model.predict_proba(test)[0][1]
        return flask.render_template('main.html',
                                     original_input={'Temperature':year,
                                                     'Humidity':procedure,
                                                     'Windspeed':drug},
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run(debug=True)
