from random import seed
from random import randrange
from csv import reader
from math import exp
 
# Cargar archivo CSV. Esta funcion convierte los datos de entrada a un objeto lista, eliminando las filas vacías 
def load_csv(archivo):
	datos = list()
	with open(archivo, 'r') as file:
		lector = reader(file)
		for fila in lector:
			#si la fila es vacia la evitamos y continuamos pegando las siguientes filas
			if not fila: 
				continue
			datos.append(fila)
	return datos
 
# Convertir el CSV de string a float. Esta funcion la hacemos debido a que nuestra funcion load_csv nos regresa la base con las columnas en strings
def str_a_float(datos, col):
	for fila in datos:
		fila[col] = float(fila[col].strip())
 
# Encontrar el máximo y mínimo de cada columna. Esta función nos ayudará más adelante a normalizar la base de datos, es decir, alterar la escala de los datos
#para que estén entre [0,1] y los coefficientes estimados no sean muy grandes o muy pequeñas
def datos_minmax(datos):
	minmax = list()
	for i in range(len(datos[0])):
		columna = [fila[i] for fila in datos]
		minimo = min(columna)
		maximo = max(columna)
		minmax.append([minimo, maximo])
	return minmax
 
# Hacer que los datos se encuentren entre 0 y 1 
def normalizar_datos(datos, minmax):
	for fila in datos:
		for i in range(len(fila)):
			#con la siguiente formula nos aseguramos de que estén entre cero y uno los valores de cada columna pues les restamos el mínimo y dividimos el valor
			#entre la longitud del rango que puede tomar la variable, asi pues aseguramos que sea un cociente cuyo numerador<denominador.
			fila[i] = (fila[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Esta función muestreará aleatoriamente sin reemplazo la base de datos 
# en el número de divisiones que queramos para validar los datos de forma cruzada 
def cross_validation(datos, num_folds):
	datos_split = list()
	datos_copia = list(datos)
	#obtenemos el tamaño de los folds según el número de folds que queremos
	fold_size = int(len(datos) / num_folds)
	for i in range(num_folds):
		fold = list()
		while len(fold) < fold_size:
			#generamos un índice aleatorio
			indice = randrange(len(datos_copia))
			#nos quedamos con el fold generado por nuestro ínidce generado aleatoriamente
			fold.append(datos_copia.pop(indice))
		datos_split.append(fold)
		#nos regresa una lista de la base de datos dividida en el número de folds que pedimos
	return datos_split
 
# Esta función calcula la precisión de las predicciones de nuestro algoritmo
def precision(real, pred):
	correct = 0
	for i in range(len(real)):
		if real[i] == pred[i]:
			correct += 1
	#nos regresa la frecuencia de exito de nuestras predicciones
	return correct / float(len(real)) * 100.0
 
# Evaluar y validar el algoritmo usando referencias cruzadas 
def eval_algoritmo(datos, algoritmo, num_folds, *args):
	folds = cross_validation(datos, num_folds)
	puntaje_precision = list()
	for fold in folds:
		#convertimos a 'base de datos' nuestra lista de folds
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		#generamos nuestro fold de entrenamiento
		for fila in fold:
			fila_copy = list(fila)
			test_set.append(fila_copy)
		#generamos nuestro vector de predicciones		
		pred = algoritmo(train_set, test_set, *args)
		#nuestro vector de tratamiento real 
		real = [fila[-1] for fila in fold]
		#obtenemos precisión del algoritmo
		accuracy = precision(real, pred)
		puntaje_precision.append(accuracy)
		#nos regresa la precisón del algoritmo para cada fold
	return puntaje_precision
 
# Hace la predicción del tratamiento
def predict(fila, coefs):
	yhat = coefs[0]
	for i in range(len(fila)-1):
		#y hat es el producto punto de nuestros controles por nuestros coeficientes 
		yhat += coefs[i + 1] * fila[i]
	#nos regresa las y gorros generadas por nuestro modelo logit.
	return 1.0 / (1.0 + exp(-yhat))
 
# Estima los coeficientes de la regresión logit utilizando el método del gradiente estocástico
def coefs_sgd(train, step, num_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	#epoch es el número de veces que nuestro algoritmo correra sobre las bases de entrenamiento para actualizar los coeficientes
	for epoch in range(num_epoch):
		for fila in train:
			#generamos las predicciones usando nuestra base de entrenamiento
			yhat = predict(fila, coef)
			#obtenemos el error de nuestra predicción
			error = fila[-1] - yhat
			#actualizamos nuestro coeficiente del intercepto condicionado al tamaño de paso que establecimos arbitrariamente
			coef[0] = coef[0] + step * error * yhat * (1.0 - yhat)
			#actualizamos nuestros coeficientes condicionado al tamaño de paso que establecimos arbitrariamente
			for i in range(len(fila)-1):
				coef[i + 1] = coef[i + 1] + step * error * yhat * (1.0 - yhat) * fila[i]
	#regresamos coeficientes actualizados por el entrenamiento			
	return coef
 
# Algoritmo de regresión logit con coeficientes derivados del método de gradiente estocástico
def logit_reg(train, test, step, num_epoch):
	prediccion = list()
	coef = coefs_sgd(train, step, num_epoch)
	for fila in test:
		yhat = predict(fila, coef)
		#debido a que nuestra predcción es de un problema de clasificación, es decir predecir si es uno (tratado) o cero (no tratado) tenemos que hacer un redondeo simple
		yhat = round(yhat)
		prediccion.append(yhat)
	return(prediccion)
 


#Probar el modelo de regresión logistica en una base de datos (seed con clave única)
seed(174064)

# Lo que se hará a continuación es evaluar la precisión del algoritmo con tres bases distintas para evaluar su precisión. 
# Las bases corresponden a una base con variables fisiológicas para predecir si personas con tumores tienen uno de la clse benigno o maligno
# Otra base con variables demográficas y de seguimiento a través de los años para saber si las personas asistieron a la universidad o no 
# Finalmente una base donde existen variables de química sanguínea para predecir si las personas tienen diabetes o no. 

#Para escoger la variables más relevantes en la base de datos de tumores lo que se hará es una selección de controles con un lasso para predicción.
#Este lasso se llama theory driven y a diferencia del lasso normal este está orientado a conseguir las mejores variables predictivas, no las mejores 
#variables explicativas.

##### BASE TUMORES 

# Cargar el archivo CSV 
archivo = 'data_tumores_procesada.csv'
datos = load_csv(archivo)
# Nos aseguramos de los datos se lean como float
for i in range(len(datos[0])):
	str_a_float(datos, i)

# Normalizamos los datos para que los datos estén entre [0,1] y que los coeficientes estimados tengan la misma escala
minmax = datos_minmax(datos)
normalizar_datos(datos, minmax)

# Correr algoritmo y definir sus parámetros, para estimar la precisión de la predicción del modelo 
num_folds = 5 
step = 0.1
num_epoch = 100
puntaje_precision = eval_algoritmo(datos, logit_reg, num_folds, step, num_epoch)
print('Puntajes de precisión de los k-folds: %s' % puntaje_precision)
print('Precisión promedio del algoritmo en base de tumores: %.3f%%' % (sum(puntaje_precision)/float(len(puntaje_precision))))

##### BASE DIABETES 

# Cargar el archivo CSV 
archivo = 'datos_diabetes.csv'
datos = load_csv(archivo)
# Nos aseguramos de los datos se lean como float
for i in range(len(datos[0])):
	str_a_float(datos, i)

# Normalizamos los datos para que los datos estén entre [0,1] y que los coeficientes estimados tengan la misma escala
minmax = datos_minmax(datos)
normalizar_datos(datos, minmax)

# Correr algoritmo y definir sus parámetros, para estimar la precisión de la predicción del modelo 
num_folds = 5 
step = 0.1
num_epoch = 100
puntaje_precision = eval_algoritmo(datos, logit_reg, num_folds, step, num_epoch)
print('Puntajes de precisión de los k-folds: %s' % puntaje_precision)
print('Precisión promedio del algoritmo en base de diabetes: %.3f%%' % (sum(puntaje_precision)/float(len(puntaje_precision))))

### BASE DE EDUCACION SUPERIOR

# Cargar el archivo CSV 
archivo = 'data_edusup.csv'
datos = load_csv(archivo)
# Nos aseguramos de los datos se lean como float
for i in range(len(datos[0])):
	str_a_float(datos, i)

# Normalizamos los datos para que los datos estén entre [0,1] y que los coeficientes estimados tengan la misma escala
minmax = datos_minmax(datos)
normalizar_datos(datos, minmax)

# Correr algoritmo y definir sus parámetros, para estimar la precisión de la predicción del modelo 
num_folds = 5 
step = 0.1
num_epoch = 100
puntaje_precision = eval_algoritmo(datos, logit_reg, num_folds, step, num_epoch)
print('Puntajes de precisión de los k-folds: %s' % puntaje_precision)
print('Precisión promedio del algoritmo en base de educación superior: %.3f%%' % (sum(puntaje_precision)/float(len(puntaje_precision))))


