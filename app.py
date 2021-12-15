from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



app = Flask(__name__)

@app.route("/", methods = ["GET"])
def hello_world():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def preds():
    vx1 = request.form["x1"]
    vx2 = request.form["x2"]
    vx3 = request.form["x3"]
    vx4 = request.form["x4"]
    vx5 = request.form["x5"]
    vx6 = request.form["x6"]
    vvx7 = request.form["x7"]
    vvx8 = request.form["x8"]

    vx7 = []
    
    if vvx7 =='O3':
        vx7=0
    if vvx7 == 'PM10':
        vx7=1
    if vvx7 =='PM25':
        vx7=2
    if vvx7 == 'PM26':
        vx7=3

    vx8 = []
    
    if vvx8 =='DKI2':
        vx8=0
    if vvx8 == 'DKI3':
        vx8=1
    if vvx8 =='DKI4':
        vx8=2
    if vvx8 == 'DKI5':
        vx8=3

    df1 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-januari-tahun-2021.csv')
    df2 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-februari-tahun-2021.csv')
    df3 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-maret-tahun-2021.csv')
    df4 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-april-tahun-2021.csv')
    df5 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-mei-tahun-2021.csv')
    df6 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-juni-tahun-2021.csv')
    df7 = pd.read_csv('indeks-standar-pencemar-udara-di-provinsi-dki-jakarta-bulan-juli-tahun-2021.csv')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)
    df.fillna(df.mean(), inplace = True)
    label = LabelEncoder()
    list_feature = ['critical', 'categori', 'location']
    df[list_feature] = df[list_feature].apply(label.fit_transform)
    scaler = MinMaxScaler()
    categoric_columns = df.iloc[:,6:].reset_index().drop('index', axis=1)
    d = scaler.fit_transform(df[['pm10','so2','co','o3','no2','max']])
    scaled_df = pd.DataFrame(d, columns = ['pm10','so2','co','o3','no2','max'])
    scaled_df = pd.concat([scaled_df, categoric_columns], axis=1)
    X = scaled_df.drop(['categori'], axis=1)
    y = scaled_df['categori']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)
    model = logreg.fit(X_train, y_train)

    dfbaru = df[['pm10','so2','co','o3','no2','max']]
    dfbaru.loc[len(dfbaru.index)] = [vx1,vx2,vx3,vx4,vx5,vx6]
    new = scaler.fit_transform(dfbaru)
    xt = new[212]
    testing = np.insert(xt,6,values = (vx7,vx8))
    testing = testing.reshape((1,-1))
    Y_pred = model.predict(testing)

    pred = []

    if Y_pred == 0 :
        pred = "Sedang"
    if Y_pred == 1 :
        pred = "Tidak Sehat"

    return render_template("index.html", prediksi = pred , Stasiunn = vvx8)

if __name__ == "__main__":
    app.run(debug=True)