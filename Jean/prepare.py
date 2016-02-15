
# coding: utf-8

# #Import

# In[3]:

import pandas as pd
import numpy as np
import time
import data
import imp
data = imp.reload(data)

def importer(df,Si,Ii,Fi,colsY,nTests=0,test=False):
    df = ajoutDate(df)
    colsX,S,I,F = numerote(Si,Ii,Fi)
    X,y = prepare(df,colsX,colsY)
    X,y = featureScale(S,X,y)
    #X,y = decoupeJours(X,y,nTests,colsX) #On passe à un jour
    E,X,colsX = convertirE(F,X,colsX,test=True)
    d = data.data(X,y,colsX,Ii)
    if(nTests!=0):
        idx = np.random.choice(np.arange(0,np.shape(X)[0]),nTests,replace=False)
        print(pd.DataFrame(X[idx],columns=colsX))
    return d


# #Import

# In[5]:

import datetime

def ajoutDate(df):
    #time = temp = pd.DatetimeIndex(df1['DT_local'])
    date = pd.DatetimeIndex(pd.to_datetime(df['Date'], format="%Y%m%d"))
    df['MONTH'] = date.month
    df['DAY'] = date.day
    df['YEAR'] = date.year
    df['DOY'] = date.dayofyear
    #Version arrondie à la minute de la ligne du dessous : num = pd.to_datetime(pd.Series.round(df['hh(UTC)']*3600)*1000000000)
    hh = pd.DatetimeIndex(pd.to_datetime(df['hh(UTC)']*3600*1000000000))
    df['HOUR'] = hh.hour
    df['MINUTE'] = hh.minute

    #print(df.corr()["P1(W)"])
    min(df['SZA(deg)'])
    return df


# # Traitement Paramètres

# In[14]:

def numerote(Si,Ii,Fi,test=False):
    colsX = np.ndarray.tolist(np.concatenate((Si[:,0],Ii[:,0],Fi[:,0])))#to list parce que sinon la fonction index existe pas
    S = np.asarray([np.arange(0,len(Si))])
    I = np.asarray([np.arange(len(Si),len(Si)+len(Ii))])
    F = np.array(Fi, copy=True)
    F[:,0] = np.arange(len(Si)+len(Ii),len(Si)+len(Ii)+len(Fi))
    F = F.astype(int)
    if(test):
        print("=> numerote")
        print("F", F)
    return colsX,S,I,F


# #Préparation des données

# In[7]:

import sklearn.cross_validation as cv
import sklearn.preprocessing as pp #pour les scaling


# In[8]:

def prepare(df,colsX,colsY,test=False):
    #Split the treated data between X and y
    Z = vireNan(df[colsX+colsY])
    deb = 0
    sep = len(colsX)#separteur entre les X et les Y
    fin = sep+len(colsY)
    X=Z[:,deb:sep]
    y=Z[:,sep:fin]
    return X,y

import numpy as np
def vireNan(Z) :
    # Load the dataset
    Z = np.array(Z).astype(np.float)
    #Remove nan and infinite values
    masknan = ~np.any(np.isnan(Z), axis=1)
    Z = Z[masknan]
    maskfin = np.any(np.isfinite(Z), axis=1)
    Z = Z[maskfin]
    return Z

def vireZero(Z1,Z2) :
    #Z2 fait office de valeur de fref
    Z1 = np.array(Z1).astype(np.float)
    Z2 = np.array(Z2).astype(np.float)
    mask = ~np.any(Z2 == 0, axis=1)
    return Z1[mask],Z2[mask]


# 
# X,y = prepare(df,colsX,colsY)

# #Feature scale

# In[9]:

#variable globales
fsX = pp.StandardScaler()
fsy = pp.StandardScaler()

def featureScale(S,Z,z, test=False) :
    #Il faut virer les valeurs extensives de la conversion
    cols = S[:,0]
    #feature scaling inti sur X_train
    fsX.fit(Z[:,cols])
    return fsOne(cols,Z,fsX),z

def fsOne(cols,Z,fs):
    Z_fs = fs.transform(Z[:,cols])
    Z_other = np.delete(Z,cols,1)#along axis 1
    #for c in cols :
    #    Z_fs = np.insert(Z_fs,c,Z_other[])
    return np.concatenate((Z_fs,Z_other),1)


# #Découpe

# In[10]:

def decoupeMelange(X,y) :
    X_train, X_1, y_train, y_1 = cv.train_test_split(X, y, test_size=0.4, random_state=0)
    X_cv, X_test, y_cv, y_test = cv.train_test_split(X_1, y_1, test_size=0.5, random_state=0)
    return [X_train,X_cv,X_test],[y_train,y_cv,y_test]

def decoupe(X,y) :
    n = len(X)
    a = np.floor(n*0.9)
    #X_train, X_1, y_train, y_1 = cv.train_test_split(X, y, test_size=0.4)
    #X_cv, X_test, y_cv, y_test = cv.train_test_split(X_1, y_1, test_size=0.5)
    return [X[0:a],X[a:n]],[y[0:a],y[a:n]]

def decoupeJours(X,y,nTests,cols, test=False) :
    n = len(X)
    a = np.floor(n*0.6)
    
    Date = X[:,cols.index('Date')]
    days = np.unique(Date)#évite les doublons #choix date car pluriannuel
    days = np.random.choice(days,nTests, replace=False)#replace = tirage unique ou pas
    days.sort()
    
    mask = np.ones(np.shape(X)[0])
    
    Xs = [[]]#On garde de la place pour le premier
    ys = [[]]
    
    for d in days  :
        Xs.append(X[Date==d])
        ys.append(y[Date==d])
        #puis on vire de X : (du coup on le garde en fait)
        #X = X[Date!=d]
        #y = y[Date!=d]
        Date = X[:,cols.index('Date')]#on adapte aussi
        
    Xs[0] = X
    ys[0] = y
    
    if(test):
        print("forme X avant", np.shape(X))
        print("forme X après",X[0].shape)
    return Xs,ys


# X,y = decoupeJours(X,y,nJoursTest,colsX)

# In[11]:

#Convertir


# In[12]:

#Conversion données extensives
from sklearn import preprocessing
import copy
    
def convertirE(F,X_,colsX_,simple=True,test=False):#
    #X_ = np.array(Z,copy=True)
    F_ = np.array(F,copy=True)
    #colsX_ = copy.copy(cols)
    E = []
    if test :
        print("=> convertirE")
        print("F_", np.shape(F_))
        print(F_)
        print(" ")
    
    for i in range(0,np.shape(F_)[0]) : #sur les ensembles E
        ###definitions
        #on prend le premier élément de E (car on vide au fur et à mesure)
        first = 0
        e = F_[first]
        F_ = F_[1:]
        F_[:,0] -= 1 #on a viré une colonne
        col = int(e[0]) 
        deb = e[1]
        fin = e[2]
        pas = e[3]
        sigma = e[4]
        
        ###Changement des noms
        nomCol = colsX_.pop(col)
        E.append(nomCol)
        #print(nomCol)
        m = int(np.floor((fin-deb)/pas))+1#col
        if(test):
            print("m",m)
        
        colsy = [None]*m #crée une liste de 0 de taille m
        for l in  range(0,m):
            colsy[l] = nomCol+str(deb+l*pas)
        colsX_ += colsy
        
        if(test):
            print("e", e)
        
        ###conversion de l'ensemble
        #for s in range(0,np.shape(X_)[0]):
        #print(X_[s].shape)
        Y = X_[:,col]
        n = np.shape(Y)[0]#lignes
            
        X_ = np.delete(X_, col, axis=1)
        #Calcul
        idxArray = ((Y-deb)//pas).astype(int) #calcul du bon index

        masque  = np.zeros((n,m))
        masque[np.arange(n),idxArray]  = idxArray

        delta = np.zeros((n,m))
        delta[:,:] = np.arange(m)#mêmes lignes de 0 à m-1
        idx = np.ones((n,m))*np.reshape(((Y-deb)//pas).astype(int),(n,1))
        delta = np.fmin(abs(delta - idx),-abs(delta-idx)+m)*pas
        #                                          l'idx de la variable
        newCols = gaussienne(delta,sigma)
        if(test):
            print("nomCol",nomCol)
            print("n",n,"m", m, "pour fin ",fin,"deb", deb, "pas", pas)
            print("Shape X", np.shape(X_))
        
        
        X_ = np.append(X_,newCols , 1)#(1) pour l'axe "col"
    return E,X_,colsX_

def gaussienne(M,sigma):
    return np.exp(-(np.divide(np.square(M),(sigma^2))))
#def convertirE(F,X_train,X_cv,X_test,cols):
#    return convertirE(F,X_train)


# X,y = featureScale(S,X,y)

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



