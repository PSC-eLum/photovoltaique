
# coding: utf-8

# In[1]:

import numpy as np
import time
np.set_printoptions(threshold=10)#np.nan


# In[2]:

#df = pd.read_csv("DonneesMeteo - 1an.csv",delim_whitespace=True, low_memory=False, skiprows=1)
tps = time.time()
import pandas as pd
df = pd.read_excel("DonneesMeteo - 1an.xls", skiprows=1)
print(time.time() - tps)


# # Traitement Data

# In[115]:

#Paramètres
colsY = ["P1(W)", "P2(W)"]
colsY = ["P1(W)"]
#pas de doublon sinon algo transo perdu
nTests = 0
d = 5
#Valeurs à scaler : au début
Si = np.array([
    ["IrrPOA(W/m2)"],
    ])
#Valeurs inutiles : pour classer
Ii = np.array([
    ["Date"],
    ["hh(UTC)"],
    ["DOY"]
    ])
#Valeurs extensives : trier par ordre apparition, attention, c'est F et plus E
Fi = np.array([
    ["DOY",0,367,10], 
    ["hh(UTC)",0,24,3],
    ["SZA(deg)",int(min(df['SZA(deg)'])),int(max(df['SZA(deg)'])),10]
    ])


# In[116]:

import prepare
X,y,colsX,S,I,F = prepare.importer(df,Si,Ii,Fi,colsY)
test(X,colsX)


# # Boulot

# In[143]:

#Conversion données extensives
from sklearn import preprocessing
import copy
    
def convertirE(G,Z,cols,simple=True,test=False):#!Z
    X_ = np.array(Z,copy=True)
    F_ = np.array(G,copy=True)
    colsX_ = copy.copy(cols)
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
        #print(nomCol)
        m = int(np.ceil((fin-deb)//pas))+1#col
        
        colsy = [None]*m #crée une liste de 0 de taille m
        for l in  range(0,m):
            colsy[l] = nomCol+str(deb+l*pas)
        colsX_ += colsy
        
        
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
        delta = np.fmin(abs(delta - idx),(delta-idx)%m)
        #                                          l'idx de la variable
        
        newCols = delta
        #newCols = gaussienne(delta,sigma)
        plt.plot(np.arange(n),newCols)
        #tab

        #print("MAX",Y[m = (fin-deb)//pas+1])
        #print(s,np.argmax(idxArray),Y[np.argmax(idxArray)],X[np.argmax(idxArray)])
        if(test):
            print("nomCol",nomCol)
            print("n",n,"m", m, "pour fin ",fin,"deb", deb, "pas", pas)
        X_ = np.append(X_,newCols , 1)#(1) pour l'axe "col"
    return F_,X_,colsX_

def gaussienne(M,sigma):
    return np.exp(-(np.square(M)//(sigma^2)))
#def convertirE(F,X_train,X_cv,X_test,cols):
#    return convertirE(F,X_train)


# In[147]:

#Conversion
G = np.array([
        [50],
        [12],
        [20]
    ])
serie = 0
E = np.insert(F,3,G[:,0],axis=1)#insérer en 3 ème pos
print(E)
Ea,Xa,colsXa = convertirE(E,X,colsX,test=True)
test(Xa,colsXa)


# In[138]:

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(10, 5, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


# In[106]:

n=3
m=4
idx = [1,1,3]
print(idx)
row = np.arange(m)
print(row)
M = np.zeros((n,m))
M[np.arange(n),idx] = idx
print(M)
M[:,:] = row
print(M)
plt.show()


# In[85]:

#Test conversion E SANS IMPACT
G = np.array([
        [100],
        [12],
        [20]
    ])
serie = 0
print(np.shape(X[0]))

E = np.concatenate([F,G],axis=1)
Ea,Xa,colsXa = convertirE(E,X,colsX)
idx = np.random.choice(np.arange(0,np.shape(X[serie])[0]),4,replace=False)

Xpd = pd.DataFrame(X[serie][idx],columns=colsX)
print(Xpd)

Xpd = pd.DataFrame(Xa[serie][idx],columns=colsXa)
print(Xpd)
print(I)
print("Xa",np.shape(Xa[0]))
print("Xa d",np.shape(np.delete(Xa[0],I,axis=1)))


# In[ ]:

#fonctions communes à toutes les régressions
from sklearn.ensemble import RandomForestRegressor
import math

def rms(Y1,Y2):
    Y1 = np.reshape(Y1,(len(Y1),1))
    Y2 = np.reshape(Y2,(len(Y2),1))
    rms = np.sqrt(np.mean((Y1- Y2)**2,axis=0))#mettre axis=0 p=quand y1, y2
    #print("rms", rms.shape, "shapes", Y1.shape, Y2.shape)
    return rms#je prends qu'une sortie

def RF(n,I_,X_,y_,poids=1):
    tps = time.time()
    Z = np.delete(X_[0],I_,axis=1)
    z = y_[0]
    #print(Z[0])
    clf = RandomForestRegressor(n_estimators=n)
    if poids.all() ==1 : #ca veut dire qu'il faut virer toutes les valeurs des test set du train set
        #A FAIRE : ci dessus
        clf.fit(Z,z)
    
    y_pred = [clf.predict(Z[0])]#dans tous les cas au moins un train
    RMS   = np.zeros((len(X_),1)) #je prends qu'une sortie
    RMS[0]= rms(y_pred[0],y_[0])
    for i in np.arange(1,len(X_)):
        if poids.all()!=1 :
            clf.fit(Z,z,sample_weight=poids[i])
        y_pred.append(clf.predict(np.delete(X_[i],I,axis=1)))
        RMS[i] = rms(y_pred[i],y_[i])
        #regroupe data       #prédire     #valeur intéressante
    print("RF :",time.time()-tps,"s")
    return y_pred,RMS    


# In[6]:

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


#Plotter un test
def plotTest(colsX,X,y,y_):    
    colors = cm.rainbow(np.linspace(0, 1, len(days)))
    Z = np.transpose(np.vstack((X,y_reel,y_pred)))
    for d,c in zip(days,colors)  :
        z = Z[Date==d]
        z = z[z[:,0].argsort()]
        #print(np.shape(z))
        c_pred    = plt.plot(z[:,0],   z[:,1]  , label=(str(int(d))+'(préd)'), color=c, linestyle=':')
        c_reel    = plt.plot(z[:,0],   z[:,2]  , label=(str((d))+'(eff)'), color=c,linestyle=style)
        
    plt.xlabel('heure')
    plt.ylabel('P(W)')
    #plt.subplot(121)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)
    return days

def plotComp(param,valeurs,Z,cols,axeX="param",axeY="valeurs",row=1):
    colors = cm.rainbow(np.linspace(0, 1, len(Z)))
    plt.subplot(row,1,1)
    plt.plot(param,valeurs[0],color=colors[0],label="train",linestyle="--")
    for k in np.arange(1,len(valeurs)) :
        plt.plot(param,valeurs[k],color=colors[k],label=str(Z[k][0,cols.index('Date')]))

    plt.xlabel(axeX)
    plt.ylabel(axeY)
    #plt.subplot(121)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)

def plotRMS(RMS,c,nom=""):
    c_ = [plt.plot(RMS[:,0],RMS[:,1], label=(nom+"(train)"),color=c, marker="v")]
    #dans tous les cas au moins un train
    for i in np.arange(1,len(RMS)):
        c_.append(plt.plot(RMS[:,0],RMS[:,1+i], label=("ex"+str(i)),color=c , marker="o"))
        
def test(X,colsX,nTests=3):
    if(nTests>0):
        idx = np.random.choice(np.arange(0,np.shape(X)[0]),nTests,replace=False)
        print(pd.DataFrame(X[idx],columns=colsX))
    return


# #test précision de la conversion
# G = np.array([
#         [10,30,100,300,367],
#         [1,3,8,12,24],
#         [3,6,10,20,38]
#     ])
# #PARAMS
# precision = np.mean(np.divide(np.transpose([F[:,2]-F[:,1]]),G),axis=0)
# print(precision.shape)
# 
# #VAEURS
# #On veut tracer RMS selon précision pas : 1 couleur par exemple
# RMS = np.zeros((len(X),len(precision)))
# for k in np.arange(0,len(precision)) :#pour chaque exemple
#     #On crée le E associe et on converti
#     g = np.transpose([G[:,k]])
#     E = np.concatenate([F,g],axis=1)
#     #On compute, #attention on se refait toujours le même
#     Ea,Xa,colsXa = convertirE(E,X,colsX)
#     y_, RMS_ = RF(d,I,Xa,y)#RMS : vecteur même taille pour chaque couple pas 
#     RMS[:,k] = np.transpose(RMS_)#pour ajouter en tant que ligne
#     
# #PLOT
# print(RMS.shape)
# plotComp(precision,RMS,X,colsX,axeX="Precision",axeY="RMS")

# In[ ]:




# In[ ]:

#Mise ne place des poids
import time
def calculeDelta(Z,cols):
    D = [[]]
    for k in np.arange(1,len(Z)):
        dates = pd.to_datetime(Z[0][:,cols.index('Date')]   , format="%Y%m%d")
        fin  = pd.to_datetime(Z[k][0,cols.index('Date')], format="%Y%m%d")
        tps = time.time()
        delta = np.array((fin-dates).days, dtype=np.float)#float pour infini   
        print("calculeDelta a pris :", time.time() - tps, "s")
        D.append(delta)
    return D

def calculePoids(Z,D,cols,l=7,t=7,m=1,M=10,ap=0):
    poids_ = np.zeros((len(Z),len(Z[0])))
    poids_[0] = np.ones((1,len(Z[0])))#pour le training il faut un zero pour que ca compte pas
    for k in np.arange(1,len(Z)):
        delta = D[k]
        #virer ce qui est apres
        delta[(-l<=delta)&(delta<0)] = np.inf
        #donnees ultérieures
        if ap == 0 :
            delta[delta<l] = np.inf
        elif ap == 1 :
            delta[delta<l] = abs(delta[delta<l])
        #delta[delta<0] = np.inf
        poids_[k] = m+(M-m)*np.exp(-np.true_divide(delta,t))#valeur minimale
    return poids_

def combine(tab):
    rep = [[0]]
    for k in np.arange(0,len(tab)) :#pour chaque paramètre
        tmp = []
        for v in tab[k] : #pour chaque valeur du paramètre
            for l in rep : #pour chaque situtaion déjà faite
                tmp.append(l+[v])
        rep = tmp
    return np.array(rep)[:,1:]  


# #Observer l'effet de poids
# s = 1
# print(s)
# poids = calculePoids(X,colsX)
# X_ = np.arange(0,len(X[0]))
# plotComp(X_,poids, X, colsX,axeX="exemple",axeY="poids")
# #print("x", X_.shape, "y", Y_.shape)

# In[21]:

D = calculeDelta(X,colsX)


# In[22]:

#La courbe de prédiction
#PARAMS
val = np.array([#valeurs prises pour chaque param
        [50],
        [10,50],
        [1],
        [10],
        [0]
    ])
p = combine(val)
#print(p)

indices = np.arange(0,len(p))

#VAEURS
#On veut tracer RMS selon précision pas : 1 couleur par exemple
RMS = np.zeros((len(X),len(p)))
for k in np.arange(0,len(p)) :#pour chaque exemple
    poids = calculePoids(X,D,p[k,0],p[k,1],p[k,2],p[k,3],p[k,4])
    y_, RMS_ = RF(d,I,X,y,poids)#RMS : vecteur même taille pour chaque couple pas 
    RMS[:,k] = np.transpose(RMS_)#pour ajouter en tant que ligne


# In[ ]:


#PLOT
print(RMS.shape)
print(poids.shape)
plt.plot(np.arange(len(poids)),poids)
plt.show()
plotComp(indices,RMS,X,colsX,axeX="Indices",axeY="RMS",row=2)
plt.savefig("abs")
plt.show()

#FARIE : tracer du one vs all pour chaque variable


# In[ ]:

#Comaprer avec/sans E : valable plus que pour le 1er exemple
Date = X[1][:,colsX.index('Date')]
days = np.unique(Date)#évite les doublons #choix date car pluriannuel
days = np.random.choice(days,5, replace=True)#replace = tirage unique ou pas
X1 = X[1][:,colsX.index('hh(UTC)')]
X2 = Xa[1][:,colsX.index('hh(UTC)')]
y0 = y[1][:,0]
y1,inutile = RF(d,I,X,y)
print(y1[1].shape)
y1 = y1[1][:,0]
y2,inutile = RF(d,I,Xa,y)
y2 = y2[1][:,0]
c = cm.rainbow(np.linspace(0, 1, 3*len(days)))

print("X1", X1.shape, "y1", y1.shape, "y0", y0.shape)
Z1 = np.transpose(np.vstack((X1,y1,y0)))
Z2 = np.transpose(np.vstack((X2,y2,y0)))
plt.clf()
for d,k in zip(days,np.arange(0,3*len(days)))  :
    z = Z1[Date==d]
    z1 = z[z[:,0].argsort()]
    z = Z2[Date==d]
    z2 = z[z[:,0].argsort()]
    #print(np.shape(z))
    c_pred     = plt.plot(z1[:,0],   z1[:,1]  , label=(str(int(d))+'(préd)'), color=c[3*k], linestyle='-')
    c_predE    = plt.plot(z2[:,0],   z2[:,1]  , label=(str(int(d))+'(préd+conv)'), color=c[3*k+1], linestyle='-')
    c_reel    = plt.plot(z1[:,0],   z1[:,2]  , label=(str((d))+'(reel)'), color=c[3*k+2],linestyle='--')

plt.xlabel('heure')
plt.ylabel('P(W)')
#plt.subplot(121)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)
    

plt.show()


# In[ ]:

plt.show()


# In[ ]:




# In[ ]:




# In[ ]:

def RFoverDepth(nArray,X,y,c):
    #Conversion
    nArray = np.array(nArray).astype(np.integer)
    #tableau avec les valeurs d'erreur
    RMS=np.zeros((nArray.size,1+len(X)))
    
    i=0
    for n in nArray:
        y_=RF(n,X,y)#2 = col du test
        RMS[i,0]= n
        for j in np.arange(0,len(X)) :
            RMS[i,1+j]= rms(y_[j] ,y[j])#1+j car n
        i+=1
        print("n ",n)
    plotRMS(RMS,c)

    
    
    return RMS


# In[ ]:

#Test Depth
G = np.array([
        [10],
        [0.25]
    ])
E = np.concatenate([F,G],axis=1)
Ea,Xa,colsXa = convertirE(E,X,colsX)
print("Sans E")
print(RFoverDepth(np.arange(1,10,1),X,y,"black"))
print("Avec E")
print(RFoverDepth(np.arange(1,10,1),Xa,y,"green"))
plt.show()


# In[ ]:


def RFoverE(EpasArray,n, X_train, X_cv, y_train, y_cv,c):
    pasArray = np.array(pasArray)
    RMS=np.zeros((pasArray.size,3))
    i=0
    for pas in pasArray:
        print("    ",pas)
        #lb = labelBinarize(deb,fin,pas)
        #Transformation data
        X_train_ =convertirExtensifEnPlace(nCol,deb,fin,pas,X_train)
        X_cv_    =convertirExtensifEnPlace(nCol,deb,fin,pas,X_cv)
        y_train_, y_cv_= randomforest(n,X_train_,y_train,X_cv_)
        RMS[i,0]= pas
        RMS[i,1]= rms(y_cv_    ,y_cv)
        RMS[i,2]= rms(y_train_ ,y_train)
        i+=1
    plotRMS(RMS,c)
    return RMS


# In[ ]:




# n=5
# #Cross sur les differentes extensions
# #d'abord DOY
# deb =0
# fin =366
# pasArray = [1,3,10,30,100]
# pasArray = [1]
# randomforestcrossoverext(nDOY,deb,fin,pasArray,n, X_train, X_cv, y_train, y_cv,c)
# print("fin")
# plt.show()

# n=5
# #Cross sur les differentes extensions
# #puis les hh
# deb =0
# fin =24
# pas = [0.1,0.4,7.8,1,3,6,9,12]
# randomforestcrossoverext(nhh,deb,fin,pas,n, X_train, X_cv, y_train, y_cv,c)

# #Couleurs
# import matplotlib.cm as cm
# colors = cm.rainbow(np.linspace(0, 1, len(nArray)))
# 
# for c,n in zip(colors,nArray) :
#     print("n ",n)
#     randomforestcrossoverext(nDOY,deb,fin,pas,n, X_train, X_cv, y_train, y_cv,c)
# print("fin")
# plt.show()

# In[583]:

n=5
#Croisement entre les hh et DOY
E = [
    ["hh",  nhh ,0,24, [6,1,0.3,0.1]],#hh
    ["DOY", nDOY,0,366,[30,10,3,1]]#DOY
    ]


print(E)
#Code
b1=1
b2=0

import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(E[b1][4])))

print(E[b1][0])
i=4
for c,pas1 in zip(colors,E[b1][4]) :
    print(pas1)
    X_train_ = convertirExtensifEnPlace(E[b1][1],E[b1][2],E[b1][3],pas1,X_train)
    X_cv_    = convertirExtensifEnPlace(E[b1][1],E[b1][2],E[b1][3],pas1,X_cv)
    point = plt.scatter(-1,i,color=c)
    #print("shape X_train", np.shape(X_train_))
    #print("shape X_train", np.shape(X_cv_))
    
    E[b2][1] -= 1#attention là on a viré une colonne, il faut changer nDOY
    print("  ",E[b2][0])
    
    #pas de nouvelles boucle
    print("  ",pas2)
    randomforestcrossoverext(E[b2][1],E[b2][2],E[b2][3],E[b2][4],n, X_train_, X_cv_, y_train, y_cv,c)
    i+=5
print("fin")
plt.show()


# In[508]:

#prioriser la data


# In[327]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[504]:

# Plot outputs 2D
import matplotlib.cm as cm

def plotterIterations(X,nX,nIt,y,ny) :
    #On se ramène à des journées
    EltIteration = X[:,nIt]
    Iterations = np.unique(EltIteration)
    print(Iterations)

    #Gérer plusieurs couleurs
    colors = cm.rainbow(np.linspace(0, 1, len(Iterations)))
    
    i = 0
    for k,c in zip(Iterations,colors)  :
        #On prend que le jour
        condition = EltIteration == k #entre 60 et 153
        xPlot = np.extract(condition, X[:,nX])
        yPlot = np.extract(condition, y[:,ny])
        plt.scatter(xPlot,yPlot, color=c,s=2)
        if i==10 :
            break
        i+=1
        #sort
        #print(np.concatenate(xPlot,yPlot))
        #z = np.sort(np.concatenate(xPlot,yPlot),0)
        #plt.plot(z[:,0], z[:,1], color=c,linewidth=2)
        #print(k)
        #print(xPlot)
        #print(yPlot)
        #print(z[:,0])
        #print(z[:,1])
        #break

    #plt.plot(X_test, regr.predict(X_test), color='blue',linewidth=3)
    plt.xticks(())
    plt.yticks(())

    plt.title('')


# In[18]:




# In[759]:

a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
b = a+10
c = np.hstack((a,b))
d= c[0]
d[1:]


# In[555]:




# In[ ]:



