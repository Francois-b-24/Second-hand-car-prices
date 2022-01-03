""" Librairie pour les prédictions """


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler,RobustScaler
from sklearn.metrics import median_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
from IPython.display import display
from ipywidgets import interactive
import ipywidgets as widgets
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider

foret_=joblib.load('foret.pkl')

def preprocess():
    """Initialisation des paramètres initiaux de preprocessing pour tous les modèles. C'est-à-dire sans modification des hyperparamètres"""

    quanti = ['capacite', 'cO2', 'kilometrage','portes','date','CV']
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer()),
        ('scaler', RobustScaler())])

    quali = ['transmission', 'energie']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value='Donnee_manquante')),
        ('numérisation', OneHotEncoder(handle_unknown = 'ignore',sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, quanti),
            ('cat', categorical_transformer, quali)])
    return preprocessor



def pipeline(preprocessor, X_train, y_train):
    """Pipeline pour l'estimation de tous les modèles sans optimisation des hyperparamètres."""
    preprocessor=preprocess()
    resultat={}
    fit = []
    modele=[LinearRegression(),Ridge(),Lasso(),ElasticNet(),KNeighborsRegressor(), 
            RandomForestRegressor(random_state=10), SVR(), TheilSenRegressor()]
    for i in range(len(modele)):
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', modele[i])])

        scores = cross_val_score(pipe, X_train, y_train, cv=10)
        resultat[modele[i]] = [round(scores.mean(),3),round(scores.std(),3)]
        fit.append(pipe.fit(X_train, y_train))
        
    return fit,resultat



def mco(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing pour le modèle des moindres carrés ordinaires."""
    param_grid = {
    'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
    'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
    'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()]
    }
    grid_search = GridSearchCV(modele, param_grid, cv=10, return_train_score=True).fit(X_train, y_train)
    
    data=pd.DataFrame(grid_search.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['mean_train_score','std_train_score','mean_test_score','std_test_score', 'param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]



def ridge(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle Ridge."""
    param_dist = {
    'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
    'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
    'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
    'regressor__alpha': [random.expovariate(10)]
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__alpha','mean_train_score','std_train_score','mean_test_score','std_test_score', 'param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]



def lasso(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle Lasso."""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
        'regressor__alpha': [random.expovariate(10)]
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__alpha','mean_train_score','std_train_score','mean_test_score','std_test_score', 'param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]



def elasticnet(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle Lasso."""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
        'regressor__alpha': [random.expovariate(10)],
        'regressor__l1_ratio': [random.expovariate(10)]
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__l1_ratio','param_regressor__alpha','mean_train_score','std_train_score','mean_test_score','std_test_score','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]



def knn(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle KNN."""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
        'regressor__n_neighbors': range(3,15,1)
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort= data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__n_neighbors','mean_train_score','std_train_score','mean_test_score','std_test_score','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]


def foret(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle de la forêt aléatoire"""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
        'regressor__n_estimators' : range(100,500,100),
        'regressor__max_depth': range(0,8),
        'regressor__criterion': ['mse','mae'],
        'regressor__max_features':['auto','sqrt','log2']
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__n_estimators','param_regressor__max_depth','param_regressor__criterion','param_regressor__max_features','mean_train_score','std_train_score','mean_test_score','std_test_score','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5],Random


def svr(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour le modèle des Support Vectors Regressors"""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()],
        'regressor__C': [1e0, 1e1, 1e2, 1e3],
        'regressor__kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'regressor__gamma':['scale','auto'],
        'regressor__epsilon':[random.expovariate(10)]
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['param_regressor__C','param_regressor__kernel','param_regressor__gamma','param_regressor__epsilon','mean_train_score','std_train_score','mean_test_score','std_test_score','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]


def ts(modele, X_train, y_train):
    """Optimisation des paramètres de precessosing et les hyperparamètres pour l'estimateur de Theil Sen"""
    param_dist = {
        'preprocessor__cat__imputer__strategy': ['most_frequent','constant'],
        'preprocessor__num__scaler' :[MinMaxScaler(),RobustScaler()],
        'preprocessor__num__imputer': [SimpleImputer(strategy='median'),SimpleImputer(strategy='mean'),KNNImputer()]
    }
    Random = RandomizedSearchCV(modele, param_dist, cv=20, random_state=10, return_train_score=True).fit(X_train, y_train)
    data=pd.DataFrame(Random.cv_results_)
    data_sort = data.sort_values(by = 'mean_test_score', ascending=False)
    mycolumns = ['mean_train_score','std_train_score','mean_test_score','std_test_score','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer', 'param_preprocessor__num__scaler']
    return data_sort[mycolumns][:5]




"""Widget interactif pour choisir les caracatéristiques de son véhicule et en avoir la prédiction du prix de vente"""
def depart():
    
    a = widgets.FloatSlider(description='Kilométrage', min=0, max=50, step=0.5)

    b =widgets.Select(
        options=['Diesel', 'Essence','GPL ou GNL', 'Hydbride', 'Electrique'],
        # rows=10,
        description='Energie:',
        disabled=False)

    c = widgets.IntSlider(
        min=0,
        max=9,
        step=1,
        description='Capacité :',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    d = widgets.FloatSlider(description='cO2', min=0, max=50, step=0.5)

    e = widgets.Select(
        options=['Manuelle', 'Automatique','Semi-automatique'],
        # rows=10,
        description='Transmission:',
        disabled=False)


    f = widgets.IntSlider(
        min=2,
        max=5,
        step=1,
        description='Portes:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    g = widgets.FloatSlider(description='CV:', min=1, max=150, step=0.5)

    h = widgets.IntSlider(
        min=1960,
        max=2020,
        step=1,
        description='Date:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    return a,b,c,d,e,f,g,h

def carac(kilometrage, energie, capacite, co2, transmission, portes, cv,date):
    caracteristiques = {
    'kilometrage':[kilometrage],
    'energie':energie,
    'capacite':[capacite],
    'cO2':[co2],
    'transmission':transmission,
    'portes':[portes],
    'CV':[cv],
    'date':[date],
    }
    
    pred = foret_.predict(pd.DataFrame.from_dict(caracteristiques, orient='columns'))
    return print('Le prix auquel le véhicule doit être mis en vente est de :', round(pred[0],2), '€')

def result():
    
    a,b,c,d,e,f,g,h = depart()

    out= widgets.interactive_output(carac, {'kilometrage': a,'energie':b,
                                             'capacite':c, 'co2':d, 'transmission':e, 'portes':f,
                                            'cv':g, 'date':h})
    ui= widgets.VBox([widgets.VBox([a,b,c,d,e,f,g,h]), out])
    return ui

def widget():
    ui = result()
    
    form_item_layout = Layout(
        display='flex',
        flex_flow='row',
        justify_content='space-between'
    )

    form_items = [
        Box([Label(value=''),
             ui], layout=form_item_layout)
    ]

    mon_widget = Box(form_items, layout=Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch',
        width='80%'
    ))
    return mon_widget