import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

names = [
    "srcip",
    "srcport",
    "dstip",
    "dstport",
    "proto",
    "total_fpackets",
    "total_fvolume",
    "total_bpackets",
    "total_bvolume",
    "min_fpktl", 
    "mean_fpktls",
    "max_fpktl",
    "std_fpktl",
    "min_bpktl",
    "mean_bpktl",
    "max_bpktl",
    "std_bpktl",
    "min_fiat",
    "mean_fiat",
    "max_fiat",
    "std_fiat",
    "min_biat",
    "mean_biat",
    "max_biat",
    "std_biat",
    "duration",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
    "sflow_fpackets",
    "sflow_fbytes",
    "sflow_bpackets",
    "sflow_bbytes",
    "fpsh_cnt",
    "bpsh_cnt",
    "furg_cnt",
    "burg_cnt",
    "total_fhlen",
    "total_bhlen",
    "label"
]

categoric = [
    "srcip",
    "srcport",
    "dstip",
    "dstport",
    #"proto",
    "label"
]

notconsidered = [
    "proto",
    "min_fpktl", 
    "mean_fpktls",
    "max_fpktl",
    "std_fpktl",
    "min_bpktl",
    "mean_bpktl",
    "max_bpktl",
    "std_bpktl",
    "sflow_fpackets",
    "sflow_fbytes",
    "sflow_bpackets",
    "sflow_bbytes",
    "fpsh_cnt",
    "bpsh_cnt",
    "furg_cnt",
    "burg_cnt",
    "total_fhlen",
    "total_bhlen"
]

def reescalar(col):
    col_cod = pd.Series(col, copy=True)
    minimo = col_cod.min()
    maximo = col_cod.max()
    col_cod = (col_cod - minimo)
    col_cod = col_cod / (maximo-minimo)
    return col_cod

def recodificar(col, nuevo_codigo):
    col_cod = pd.Series(col, copy=True)
    for llave, valor in nuevo_codigo.items():
        col_cod.replace(llave, valor, inplace=True)
    return col_cod

class MiModeloPredictor():
    
    def __init__(self, modelo, test_size=0.3):
        
        # Definir nombre de archivo y modelo
        self.__archivo = "labeled-dataset.csv"
        self.__modelo=modelo
        
        # Semilla
        self.__random_state = 0
        
        # Leer los datos
        self.__datos = pd.read_csv(self.__archivo, header=None)
        self.__datos.columns = names
        
        # Definir mapeo de variable a predecir
        self.__mapeo = {'normal':0,'botnet':1}
        
        # Recodificar la variable label (Variable a predecir)
        self.__datos["label"] = recodificar(self.__datos["label"], self.__mapeo)
        
        # Eliminar Variables No útiles
        del self.__datos["srcip"]
        del self.__datos["dstip"]
        for lb in notconsidered:
            del self.__datos[lb]
        
        # Imprimir Valores y Estadísticas de los datos
        #print(self.__datos.head())
        #print(self.__datos.describe())
        #print(self.__datos.info())
        #print(self.__datos.shape)
        
        # Borrar las filas que contengan valores NA
        self.__datos.dropna(inplace=True)
        
        # Definir las variables predictoras
        self.__X = pd.DataFrame(self.__datos.iloc[:,0:23])

        # Definir la variable a predecir ("label")
        self.__y = self.__datos.iloc[:,23:]
        self.__y = pd.Series(self.__y.values.ravel())
        
        # Definir la instancia del modelo para entrenamiento
        if modelo == 'arbol':
            self.__instancia_modelo = DecisionTreeClassifier(random_state=self.__random_state,min_samples_leaf=1,criterion='gini',min_samples_split=2)
        elif modelo == 'knn':
            self.__instancia_modelo = KNeighborsClassifier(n_neighbors=5,algorithm='auto',p=2)
        elif modelo == 'KNN':
            self.__instancia_modelo = KNeighborsClassifier(n_neighbors=7,algorithm='auto',p=2)
        elif modelo == 'DT':
            self.__instancia_modelo = DecisionTreeClassifier(random_state=self.__random_state,min_samples_leaf=2,criterion='gini',min_samples_split=5,max_features="sqrt")
        elif modelo == 'kmedias':
            self.__instancia_modelo = KMeans(n_clusters=7, max_iter = 1000, n_init=100)
        elif modelo == 'QDA':
            self.__instancia_modelo = QuadraticDiscriminantAnalysis()
        elif modelo == 'RF':
            self.__instancia_modelo = RandomForestClassifier(n_estimators=10, random_state=self.__random_state)
        elif modelo == 'RF2':
            self.__instancia_modelo = RandomForestClassifier(n_estimators=100, random_state=self.__random_state)
        elif modelo == 'ETC':
            self.__instancia_modelo = ExtraTreesClassifier(n_estimators=10, random_state=self.__random_state)
        elif modelo == 'ETC2':
            self.__instancia_modelo = ExtraTreesClassifier(n_estimators=100, random_state=self.__random_state)
        elif modelo == 'SVM':
            self.__instancia_modelo = SVC(kernel='sigmoid')
        elif modelo == 'bayes':
            self.__instancia_modelo = GaussianNB()
        elif modelo == 'LDA':
            self.__instancia_modelo = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 'auto')
        elif modelo == 'DT2':
            self.__instancia_modelo = DecisionTreeClassifier(random_state=self.__random_state,min_samples_leaf=2,criterion='entropy',min_samples_split=5,max_features="sqrt")
        elif modelo == 'ADA':
            self.__instancia_modelo = AdaBoostClassifier(n_estimators=10, random_state=self.__random_state)
        elif modelo == 'XGB':
            self.__instancia_modelo = GradientBoostingClassifier(n_estimators=10, random_state=0)
        elif modelo == 'MLP':
            self.__instancia_modelo = MLPClassifier(solver='lbfgs', random_state=self.__random_state,hidden_layer_sizes=(100, ))
        else:
             raise Exception('Este modelo no está soportado!!!')
        
        # Entrenar el modelo (10-Fold Cross Validation)
        self.__scores = []
        cv = StratifiedKFold(n_splits=10, random_state=self.__random_state, shuffle=False)
        for train_index, test_index in cv.split(self.__X,self.__y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = self.__X.iloc[train_index], self.__X.iloc[test_index], self.__y.iloc[train_index], self.__y.iloc[test_index]
            self.__instancia_modelo.fit(self.__X_train, self.__y_train)
            self.__scores.append(self.__instancia_modelo.score(self.__X_test, self.__y_test))
            
        # Definir el puntaje del modelo (~ Precisión)
        self.__score = np.mean(self.__scores)
        
        # Separar el dataset de entrenamiento y de pruebas de manera estratificada
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=test_size, random_state=self.__random_state,shuffle=True,stratify=self.__y)
        
        # Encontrar las predicciones para medir su rendimiento
        self.__prediccion = self.__instancia_modelo.predict(self.__X_test)
        
        # Encontrar la matriz de confusión
        self.__MC = confusion_matrix(self.__y_test, self.__prediccion)
        
    def predecir(self, toPredict):
        return self.__instancia_modelo.predict(toPredict)
    
    @property
    def MC(self):
        return self.__MC
    
    @property
    def score(self):
        return self.__score
    
    def getPPV(self, value):
        TP = self.MC[value][value]
        FP = self.MC.sum(axis=0)[value] - TP
        FN = self.MC.sum(axis=1)[value] - TP
        PPV = TP/(TP+FP)
        return PPV
        
    def getRC(self, value):
        TP = self.MC[value][value]
        FP = self.MC.sum(axis=0)[value] - TP
        FN = self.MC.sum(axis=1)[value] - TP
        RC = TP/(TP+FN)
        return RC
    
    def printFacts(self):
        print("Modelo: {}".format(self.__modelo))
        print("Score: {}".format(self.__score))
        for llave,valor in self.__mapeo.items():
            print("Precisión {}: {:.2f}".format(llave,self.getPPV(valor)))
            print("Recall {}: {:.2f}".format(llave,self.getRC(valor)))
        print("Matriz de Confusión:\n{}".format(self.MC))
        print("")

model = MiModeloPredictor("RF")

namesZodiac = [
    "srcip",
    "srcport",
    "dstip",
    "dstport",
    "total_fpackets",
    "total_fvolume",
    "total_bpackets",
    "total_bvolume",
    "min_fiat",
    "mean_fiat",
    "max_fiat",
    "std_fiat",
    "min_biat",
    "mean_biat",
    "max_biat",
    "std_biat",
    "duration",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
    "min_flowiat",
    "mean_flowiat",
    "max_flowiat",
    "std_flowiat",
    "fb_psec",
    "fp_psec"
]

zodiacNotConsidered = [
    "srcip",
    "dstip",
    "min_flowiat",
    "mean_flowiat",
    "max_flowiat",
    "std_flowiat",
    "fb_psec",
    "fp_psec"
]

def predict(flow):
    flowDF = pd.DataFrame([flow], columns = namesZodiac)
    for lb in zodiacNotConsidered:
        del flowDF[lb]
    return model.predecir(flowDF)[0]