from flask import Flask, render_template, request, session, redirect, url_for, make_response,flash, get_flashed_messages, jsonify, send_file
import pandas as pd
import numpy as np
import tempfile, os, json, pickle
from io import StringIO
from scipy.stats import zscore
from ydata_profiling import ProfileReport
from pydantic_settings import BaseSettings
from pivottablejs import pivot_ui
import pygwalker as pyg
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans, \
    MeanShift, OPTICS, SpectralClustering
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, \
    ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


app = Flask(__name__)
app.secret_key = '115465X455A5'

app.config['MESSAGE_FLASH_DURATION'] = 5000 

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')
    session.clear()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    df_info = None  # Initialize to None
    session.clear()
    if request.method == 'POST':
        file = request.files['file']

        if file and file.filename.endswith('.csv'):
            file.save('data.csv')
            df = pd.read_csv('data.csv')
        elif file and file.filename.endswith('xlsx'):
            df=pd.read_excel(file)
        info_output = StringIO()
        df.info(buf=info_output)
        info_str = info_output.getvalue()
        num_duplicates = df.duplicated().sum()
        num_missing_values = df.isnull().sum().sum()
        df_info = {
            'shape': df.shape,
            'size': len(df),
            'description': df.describe().to_dict(),
            'info': info_str,
            'num_duplicates': num_duplicates,
            'num_missing_values': num_missing_values
        }

    return render_template('upload.html', df_info=df_info)



@app.route('/cleaning', methods=['GET', 'POST'])
def data_cleaning():
    columns=[]    
    df = pd.read_csv('data.csv')
    # df_table = df.to_html(classes='table table-bordered table-hover', index=False)
    columns=df.columns.to_list()
    if request.method == 'POST':
        if 'remove_duplicates' in request.form:
            df.drop_duplicates(inplace=True)
            flash('Duplicates removed successfully!', 'success')
            session['flash_message1'] = 'Duplicates removed successfully!'
        if 'remove_columns' in request.form:
            columns_to_remove = request.form.getlist('columns_to_remove')
            df.drop(columns=columns_to_remove, inplace=True)
            flash('Selected columns removed successfully!', 'success')
            session['flash_message2'] = 'Selected columns removed successfully!'
        if 'fill_all_values' in request.form:
            method_numeric_missing = request.form.get('method_numeric_missing')
            method_categoric_missing = request.form.get('method_categoric_missing')

            for column in df.columns:
                dt = str(df[column].dtype)

                if dt == 'int64' or dt == 'float64':
                    if method_numeric_missing == 'mean':
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif method_numeric_missing == 'median':
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(float(method_numeric_missing), inplace=True)
                else:
                    if method_categoric_missing == 'mode':
                        df[column].fillna(df[column].mode().iloc[0], inplace=True)
                    else:
                        df[column].fillna(method_categoric_missing, inplace=True)

            flash('Missing values filled successfully!', 'success')
            session['flash_message3'] = 'Missing values filled successfully!'
        columns=df.columns.to_list()
        df.to_csv('data.csv', index=False)
        return render_template('data_cleaning.html', columns=columns)

    
    return render_template('data_cleaning.html', columns=columns)

@app.route('/profiling')
def data_profiling():
    df = pd.read_csv('data.csv')
    profile = ProfileReport(df)
    profiling_report = profile.to_html()

    return render_template('data_profiling.html', profiling_report=profiling_report)

@app.route('/generate_profiling_report')
def generate_profiling_report():
    df = pd.read_csv('data.csv')
    profile = ProfileReport(df)
    profiling_report = profile.to_html()

    # Create a response containing the profiling report HTML
    response = make_response(profiling_report)

    # Set the content type to HTML
    response.headers['Content-Type'] = 'text/html'

    return response

@app.route('/visualization')
def data_visualization():
    df = pd.read_csv('data.csv')
    html_str=pyg.walk(df,return_html=True)
    return render_template("data_visualization.html",html_str=html_str)
@app.route('/modelling')
def data_modelling():
    df = pd.read_csv('data.csv')

    return render_template('data_modelling.html')

@app.route('/supervised_modelling')
def supervised_modelling():
    df=pd.read_csv('data.csv')
    columns=df.columns.to_list()
    table={}
    if request.method == 'POST':
        problem_type = request.form.get('problem_type')
        target_column = request.form.get('target_column')
        test_size = request.form.get('test_size')
        algorithms = request.form.getlist('algorithms')
        X = df.drop(columns=[target_column])
        y = df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        numerical_col = X.select_dtypes(include=np.number).columns
        categorical_col = X.select_dtypes(exclude=np.number).columns
        scaler = MinMaxScaler()
        ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                        remainder='passthrough')
        x_train_encoded = ct_encoder.fit_transform(x_train)
        x_test_encoded = ct_encoder.transform(x_test)
        x_train = scaler.fit_transform(x_train_encoded)
        x_test = scaler.transform(x_test_encoded)

        if problem_type=='regression':
            table = {"Algorithm": [], "MAE": [], "RMSE": [], "R2 Score": []}
            for algorithm in algorithms:
                if algorithm == "Linear Regression":
                    reg = LinearRegression()
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Linear Regression")
                    table["R2 Score"].append(r2score)
                    # pickle.dump(reg, open('LR.pkl', 'wb'))

                elif algorithm == "Polynomial Regression":
                    # degree = st.slider("Polynomial Degree", 2, 10, 2)
                    reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                    reg.fit(x_train, y_train)
                    y_pred = reg.predict(x_test)
                    mse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2score = r2_score(y_test, y_pred) * 100

                    table["RMSE"].append(mse)
                    table["MAE"].append(mae)
                    table["Algorithm"].append("Polynomial Regression")
                    table["R2 Score"].append(r2score)
                    # pickle.dump(reg, open('PR.pkl', 'wb'))

    

    return render_template('supervised_modelling.html', columns=columns,table_json=jsonify(table))

@app.route('/unsupervised_modelling')
def unsupervised_modelling():

    return render_template('unsupervised_modelling.html')


@app.route('/downlod')
def data_download():
    if not os.path.exists('data.csv'):
        return "Go To Upload File"  
    else:
        df = pd.read_csv('data.csv')
        df.to_csv('DATA.csv', index=False)
        return send_file('DATA.csv', as_attachment=True)
    return render_template('data_download.html')

d = {"LOR": "Logistic Regression", "DT": "Decision Trees", "RF": "Random Forest", "NB": "Naive Bayes",
     "SVM": "Support Vector Machines (SVM)",
     "GB": "Gradient Boosting", "NN": "Neural Networks", "QDA": "Quadratic Discriminant Analysis (QDA)",
     "AB": "Adaptive Boosting (AdaBoost)", "GP": "Gaussian Processes", "PT": "Perceptron", "KNC": "KNN Classifier",
     "RC": "Ridge Classifier", "PA": "Passive Aggressive Classifier", "EN": "Elastic Net",
     "LAR": "Lasso Regression",
     "LR": "Linear Regression", "PR": "Polynomial Regression",
     "SVR": "Support Vector Regression", "DTR": "Decision Tree Regression",
     "RFR": "Random Forest Regression", "RR": "Ridge Regression",
     "LASR": "Lasso Regression", "GR": "Gaussian Regression", "KNR": "KNN Regression", "ABR": "AdaBoost",
     "AP": "Affinity Propagation", "AC": "Agglomerative Clustering",
     "BC": "BIRCH", "DB": "DBSCAN", "KM": "K-Means", "MBK": "Mini-Batch K-Means",
     "MS": "Mean Shift", "OC": "OPTICS", "SC": "Spectral Clustering",
     "GMM": "Gaussian Mixture Model"
     }

cla_model = ['LOR', 'DT', 'RF', 'NB', 'SVM', 'GB', 'NN', 'QDA', 'AB', 'GP', 'PT', 'RC', 'PA', 'EN', 'LAR', 'KNC']
reg_model = ['LR', 'PR', 'SVR', 'DTR', 'RFR', 'RR', 'LASR', 'GR', 'ABR', 'KNR']
clu_model = ['AP', 'AC', 'BC', 'DB', 'KM', 'MBK', 'MS', 'OC', 'SC', 'GMM']

@app.route('/download_model/<model_type>/<model_name>')
def download_model(model_type, model_name):
    if model_type == "Regression" and model_name in reg_model:
        if os.path.exists(f"./{model_name}.pkl"):
            with open(f"{model_name}.pkl", 'rb') as file:
                return send_file(file, as_attachment=True, download_name=f"{model_name}.pkl")
    elif model_type == "Classification" and model_name in cla_model:
        if os.path.exists(f"./{model_name}.pkl"):
            with open(f"{model_name}.pkl", 'rb') as file:
                return send_file(file, as_attachment=True, download_name=f"{model_name}.pkl")
    elif model_type == "Clustering" and model_name in clu_model:
        if os.path.exists(f"./{model_name}.pkl"):
            with open(f"{model_name}.pkl", 'rb') as file:
                return send_file(file, as_attachment=True, download_name=f"{model_name}.pkl")

    return "Invalid model type or model name"
   





@app.route('/about')
def about_page():
    return render_template('about.html')
@app.route('/login')
def login_page():
    
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')


class Settings(BaseSettings):
    # Your settings here
    pass

if __name__ == '__main__':
    app.run(debug=True)