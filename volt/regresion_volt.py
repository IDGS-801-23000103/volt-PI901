import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datos = pd.read_csv("dataset_volt.csv")

X = datos[['distancia_cm','duracion_segundos','errores_detectados']]
y = datos['bateria_porcentaje']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train,y_train)

print("Modelo entrenado correctamente")