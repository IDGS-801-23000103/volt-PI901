import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

datos = pd.read_csv("dataset_volt.csv")

X = datos[['distancia_cm',
           'bateria_porcentaje',
           'duracion_segundos',
           'errores_detectados']]

y = datos['estado_robot']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

modelo = DecisionTreeClassifier()
modelo.fit(X_train,y_train)

print("Clasificador entrenado")