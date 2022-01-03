""" Librairie de nettoyage """ 

import pandas as pd
import numpy as np

donnees = pd.read_csv('ParuVendu.csv')

def annee(df: pd.DataFrame):
    """Transformation de la variable année"""
    annee=[]
    for i in range(len(donnees['Année'])):
        try:
            a=(donnees['Année'][i].split('\n'))[0]
            annee.append(a.split(' ')[1])
        except:
            annee.append(np.nan)
    return annee


def supp_colonnes(df: pd.DataFrame):
    "Suppression des colonnes : Unnamed: 0, Type_vendeur, Année, annee, Modele"""
    df=df.drop(columns=['Unnamed: 0','Type_vendeur','Année','annee','Modele']) 
    return df

def renommage(df: pd.DataFrame):
    """Renommage des colonnes"""
    df=df.rename(columns={'Prix':'prix',
                   'Kilométrage':'kilometrage',
                   'Energie':'energie',
                   'Capacité':'capacite',
                   'Emission_CO2':'cO2',
                   'Consommation':'consommation',
                   'Transmission':'transmission',
                   'Nombres de portes':'portes',
                   'Chevaux fiscaux':'CV'
                  })
    return df

def gestionvar(df: pd.DataFrame):
    """Transformation des variables dans le bon format"""
    df['prix']=df['prix'].str.replace('€','')
    df['prix']=df['prix'].str.replace(' ','').astype('float')
    df['capacite']=df['capacite'].str.replace('places','').astype('float')
    df['CV']=df['CV'].str.replace('CV','').astype('float')
    df['consommation']=df['consommation'].str.replace('litres / 100 km','').astype('float')
    df['cO2']=df['cO2'].str.replace('g/km','').astype('float')
    df['kilometrage']=df['kilometrage'].str.replace('km','')
    df['kilometrage']=df['kilometrage'].str.replace(' ','').astype('float')
    df['portes']=df['portes'].str.replace('portes avec hayon','')
    df['portes']=df['portes'].str.replace('portes','').astype('float')
    return df


def cat(df: pd.DataFrame):
    """Transformation de variables en variables categorielles"""
    for i in df.columns:
        if i == "energie":
            df.energie = df.energie.astype("category")    
        if i == "transmission":
            df.transmission = df.transmission.astype("category")       
    return df

def dup(df: pd.DataFrame):
    """Suppression des lignes dupliquées"""
    df.drop_duplicates()
    return df

def supp_colonne(df: pd.DataFrame):
    """Suppression de la colonne consommation"""
    df=df.drop(columns=['consommation']) 
    return df

def supp_na_partielle(df: pd.DataFrame): 
    """Suppression partielle des données manquantes des variables kilométrage et prix"""
    df=df.dropna(axis=0, subset=["kilometrage", "prix"])
    return df