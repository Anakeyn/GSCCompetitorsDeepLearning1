# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:29:18 2019

@author: Pierre
"""
##########################################################################
# GSCCompetitorsDL1
# Auteur : Pierre Rouarch - Licence GPL 3
# Classification des pages Web dans Google sur un mot clé en fonctions de caractérisitiques
# Deep Learning  sur un univers de concurrence 1 
# Utilisation d'un réseau de neurones feedforward simple pour une classification binaire
# Données enrichiees via Scraping précédemment.
#####################################################################################

###################################################################
# On démarre ici 
###################################################################
#Chargement des bibliothèques générales utiles
#Remarque installer les bibliothèques manquantes via conda install
#import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
#import scipy as sp  #pour l'analyse statistique - non utilisé.
import pandas as pd  #pour les Dataframes ou tableaux de données
import os
#scaler
from sklearn.preprocessing import StandardScaler
#Autres Scalers pas forcément utile mais peuvent être testés.
#from sklearn.preprocessing import MinMaxScaler
#fom sklearn.preprocessing import minmax_scale
#from sklearn.preprocessing import MaxAbsScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import QuantileTransformer
#rom sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import train_test_split


#for Deep Learning 
from keras import models
from keras import layers

print(os.getcwd())  #verif my path
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/MyPath"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif



#############################################################
#  Deep  Learning sur les données enrichies après scraping
#############################################################

#Lecture des données suite  à scraping ############
dfQPPS8 = pd.read_csv("dfQPPS7.csv")
dfQPPS8.info(verbose=True) # 12194 enregistrements.    
dfQPPS8.reset_index(inplace=True, drop=True) 

#Variables explicatives
X =  dfQPPS8[['isHttps', 'level', 
             'lenWebSite', 'lenTokensWebSite',  'lenTokensQueryInWebSiteFrequency',  'sumTFIDFWebSiteFrequency',            
             'lenPath', 'lenTokensPath',  'lenTokensQueryInPathFrequency' , 'sumTFIDFPathFrequency',  
              'lenTitle', 'lenTokensTitle', 'lenTokensQueryInTitleFrequency', 'sumTFIDFTitleFrequency',
              'lenDescription', 'lenTokensDescription', 'lenTokensQueryInDescriptionFrequency', 'sumTFIDFDescriptionFrequency',
              'lenH1', 'lenTokensH1', 'lenTokensQueryInH1Frequency' ,  'sumTFIDFH1Frequency',        
              'lenH2', 'lenTokensH2',  'lenTokensQueryInH2Frequency' ,  'sumTFIDFH2Frequency',          
              'lenH3', 'lenTokensH3', 'lenTokensQueryInH3Frequency' , 'sumTFIDFH3Frequency',
              'lenH4',  'lenTokensH4','lenTokensQueryInH4Frequency', 'sumTFIDFH4Frequency', 
              'lenH5', 'lenTokensH5', 'lenTokensQueryInH5Frequency', 'sumTFIDFH5Frequency', 
              'lenH6', 'lenTokensH6', 'lenTokensQueryInH6Frequency', 'sumTFIDFH6Frequency', 
              'lenB', 'lenTokensB', 'lenTokensQueryInBFrequency', 'sumTFIDFBFrequency', 
              'lenEM', 'lenTokensEM', 'lenTokensQueryInEMFrequency', 'sumTFIDFEMFrequency', 
              'lenStrong', 'lenTokensStrong', 'lenTokensQueryInStrongFrequency', 'sumTFIDFStrongFrequency', 
              'lenBody', 'lenTokensBody', 'lenTokensQueryInBodyFrequency', 'sumTFIDFBodyFrequency', 
              'elapsedTime', 'nbrInternalLinks', 'nbrExternalLinks' ]]  #variables explicatives

X.info()
y =  dfQPPS8['group']  #variable à expliquer,

##Sciikit Learn Scalers - choose one
scaler = StandardScaler() # Standard Scaler
#scaler =  MinMaxScaler()  #pas mieux
#scaler =  minmax_scale()
#scaler =  MaxAbsScaler()
#scaler =   RobustScaler()  #moins bon que standard scaler
#scaler =  Normalizer()
#scaler =  QuantileTransformer()
#scaler =  PowerTransformer()

scaler.fit(X)
X_Scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
X_Scaled.info()
#check some values
plt.hist( X_Scaled['isHttps'])
plt.hist( X_Scaled['lenTokensWebSite'])


#Manual  Scaled - same as StandardScaler
#X_Mean = X
#X_Mean -= X_Mean.mean(axis=0)
#X_ManualScaled = X_Mean
#X_ManualScaled /= X_Mean.std(axis=0)
#plt.hist( X_ManualScaled['isHttps'])
#plt.hist( X_ManualScaled['lenTokensWebSite'])
#X_Scaled = X_ManualScaled

########################################################
#on choisit random_state = 42 en hommage à La grande question sur la vie, l'univers et le reste
#dans "Le Guide du voyageur galactique"   par  Douglas Adams. Ceci afin d'avoir le même split
#tout au long de notre étude.
X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y, random_state=42)

X_train.shape
#(9145, 61)


##############################################################
#Réseau de Neurones

unitsNumber = 40  #nombre de neurones par couche  cachée #(~2/3*( 61+1))

#Define Sample Neural Network Model  with 2 hidden layers
model = models.Sequential()  #
model.add(layers.Dense(unitsNumber, activation='relu', input_shape=(61,)))
model.add(layers.Dense(unitsNumber, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #pour avoir  une probabilité en sortie


# compile the model with custom metrics
#Choose one optimizer :  rmsprop is generally a good enough choice.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#you could check
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['acc'])
#model.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['acc'])


#Fit the model to data
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))



############################################################
#Graphiques  et  résultats
history_dict = history.history
history_dict.keys()
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc'] #ce qui nous intéresse
epochs = range(1, len(acc) + 1)


#perte
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("QPPS8-FNN-Loss.png", bbox_inches="tight", dpi=600)
plt.show()

#précision
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy FNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("QPPS8-FNN-Accuracy.png", bbox_inches="tight", dpi=600)
plt.show()  #!!!!   affiche et remet à zéro => sauvegarder avant 




max(acc)  #meilleure  valeur de la précision sur le train set   0.8270092947030733
acc.index(max(acc))  #29

max(val_acc)  #meilleure  valeur de la précision de validation sur le test set  0.7448343719838681
#Meilleur que xgBoost non optimisé : 0.734 mais moins bien que KNN 0.7553
val_acc.index(max(val_acc))  #indice correspondant # 9

##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()






    
