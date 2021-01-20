import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,KFols,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score


#%% importation des donnees
weatherFrame=pd.read_csv("weather.csv",true_values=["Yes"],false_values=["No"]) #On ajoute les parametres optionnels truevalues et false_values pour pouvoir considérer les colonnes RainToday et RainTomorrow comme des booléens


#%% Examen des données
print("Nb de lignes et colonnes ",weatherFrame.shape,"\n","nb d'elements ",weatherFrame.size)
print("type de donnees ",weatherFrame.dtypes)

weatherFrame.hist(column="MaxTemp")
print("temperature max : ", weatherFrame['MaxTemp'].max())


#%%Préparation des données
#Suppression des valeurs aberrantes
def supprimeValeursAberrantes(frame, colonne, min, max):
    """Supprime les valeurs aberrantes d'une frame pour la colonne donnée. 
    Les valeurs aberrantes sont celles qui ne sont pas comprises entre le min et le max"""
    for i in range(frame.shape[0]):
        if frame[colonne][i]<min or frame[colonne][i]>max :
            frame[colonne][i]=np.nan #On supprime les valeurs aberrantes en les remplaçant par np.nan

supprimeValeursAberrantes(weatherFrame,'MaxTemp',-10,50)


weatherFrame.drop('RISK_MM',axis='columns',inplace=True) #On supprime la colonne 'RISK_MM' qui n'a pas de sens puisqu'elle sert à construire la variable cible. Le inplace=true sert à faire les modifications en place

#Suppression des colonnes manquantes
def supprimeColonnesManquantes(frame):
    """Supprime en place les colonnes où plus d'un tiers des valeurs manquent"""
    nbLignes=frame.shape[0]
    for colonne in frame:
        if frame.isna().sum()[colonne]>=nbLignes/3:#isna renvoie une DataFrame contenant des 1 à la place des valeurs nulles et des 0 à la place des valeurs non nulles
            frame.drop(colonne,axis='columns',inplace=True)#Si plus d'un tiers de la colonne est vide, on la supprime

supprimeColonnesManquantes(weatherFrame)

#Remplacement des valeurs numeriques
valeursMedianes=weatherFrame.median(axis=0,skipna=True,numeric_only=True)#calcul de la médiane sans compter les valeurs nulles et en ne prenant en compte que les valeurs numérques
weatherFrame.fillna(valeursMedianes, inplace=True) #On remplace en place les valeurs numeriques nulles

#remplacement des valeurs non-numeriques
def completevaleursQualitativesManquantes(frame):
    """Complète en place les valeurs qualitatives manquantes"""
    for colonne in frame:
        if frame.dtypes[colonne]=='O':#Si elles sont du type objet(=string)
            weatherFrame[colonne].fillna(method='pad',inplace=True)#On remplace par la première valeur non nulle suivante

completevaleursQualitativesManquantes(weatherFrame)

def quantitatif_to_numerique(frame):
    """Transforme en place les valeurs qualitatives en valeurs numériques"""
    for colonne in frame:
        if frame.dtypes[colonne]=='O':
            lc=LabelEncoder()
            colonne_np=frame[colonne].to_numpy()#On doit transformer la colonne en tableau numpy pour pouvoir appliquer fit_transform
            frame[colonne]=lc.fit_transform(colonne_np)#On remplace la colonne par les valeurs transformées (Vue de l'extérieur, cette fonction modifie donc la dataframe en place)
            
quantitatif_to_numerique(weatherFrame)

#Recalibrage des variables
def recalibrage(frame):
    """Permet de recalibrer les variables, pas de maodification en place"""
    sc=StandardScaler()
    frameNp=frame.to_numpy()
    sc.fit(frameNp)
    frame=pd.DataFrame(sc.transform(frameNp,copy=False),columns=frame.columns)#La modification en place n'est ici plus possible, on doit donc recréer une DataFrame à partir du tableau numpy en surchargeant son constructeur avec le parametre columns qui contient les noms des colonnes
    return frame
    
weatherFrame=recalibrage(weatherFrame)#On doit ici redéfinir la dataframe puisqu'on en a renvoyé une autre

#%%Recherche des corrélations
def supprimeDonneesNonCorrelees(frame,colonneAComparer, seuil):
    """Fonction permettant de calculer le coefficient de correlation. 
    Si ce coefficient est inferieur en valeur absolue au coefficient seuil, on supprime en place la colonne"""
    MatriceCorrelations=frame.corr(method='pearson')[colonneAComparer]
    for colonne,valeur in MatriceCorrelations.iteritems(): #Pandas.series.iteritems renvoie la liste des couples (index,valeurs)
        if abs(valeur)<seuil:
            frame.drop(colonne,axis='columns',inplace=True)
            
supprimeDonneesNonCorrelees(weatherFrame,'RainTomorrow',0.1)

pd.plotting.scatter_matrix(weatherFrame)


#%%extraction des donnees d'apprentissage et de test
def extraction(frame):
    """Fonction permettant de séparer les données d'apprentissage et les données test
    lorsque la dernière colonne de la DataFrame contient les données cibles"""
    X_data=frame.iloc[:,:-1].values #On transforme les données en tablaux numpy en les sélectionnant grâce à leurs indices à l'aide de la méthode iloc
    y_data=frame.iloc[:,-1].values
    y_data=np.where(y_data>=1, 1,0)#On remplace les valeurs numeriques sensées représenter des booléens par des valeurs plus cohérentes
    return X_data,y_data

X_data,y_data=extraction(weatherFrame)
X_train,X_test,Y_train,Y_test=train_test_split(X_data,y_data,train_size=0.75)



#%%Entraînement du modèle
lr = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
lr.fit(X_train,Y_train)
lr_y_predict = lr.predict(X_test)


#%%Évaluation du modèle
print("Accuracy_score : ",accuracy_score(Y_test,lr_y_predict))

#Confusion_matrix
c=confusion_matrix(Y_test,lr_y_predict)
print("True Negatives (response 0, predicted 0) : ", c[0,0])
print("False Positives (response 0, predicted 1) : ", c[0,1])
print("False Negatives (response 1, predicted 0) : ", c[1,0])
print("True Positives (response 1, predicted 1) : ", c[1,1])

print("Precision score : ",precision_score(Y_test,lr_y_predict))
print("Recall score : ",recall_score(Y_test,lr_y_predict))
print("F1_score : ",f1_score(Y_test,lr_y_predict))


#%%Amélioration de l'évaluation

#Validation croisée
kf = KFold(n_splits=5,shuffle=True)#Score de validation croisée sur 5 sous-ensembles
cvs=cross_val_score(lr,X_data,y_data, cv=5,scoring='accuracy')
print("Score de validation croisée: ",cvs)
print ("Score minimal sur les sous-ensembles : ", min(cvs))
print("Score moyen sur les sous-ensembles : ", cvs.mean())