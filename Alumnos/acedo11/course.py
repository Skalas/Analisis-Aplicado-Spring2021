import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

# Strings

print("Hello World!")
character_name = "John"
character_age = 35
isMale = True #boolean
print("There once was a man named " + character_name + ", " )

#saltar un renglon
print("Giraffe\nAcademy")
phrase = "Giraffe Academy"

#concatenation
print(phrase + " is cool")

#hacer minusculas o mayusculas
print(phrase.lower())
print(phrase.upper())

#preguntar
print(phrase.isupper())

#longitud
print(len(phrase))

#extraer el primer caracter del string
print(phrase[0])
#obs: en Python los indices empiezan en cero

#encontrar el indice en donde se encuentra un caracter (la primera ocurrencia)
print(phrase.index("i"))

#util
print(phrase.replace("Giraffe", "Elephant")) #reemplazar giraffe con elephant


# Numbers

from math import *

#modulo: divide el primer numero entre el segundo y me regresa el residuo
print(10 % 3)

my_num = 11
print(str(my_num) + " my favorite number") #convertir a string
print(pow(11, 2)) #11 al cuadrado
print(round(3.7))

#getting input from users
name = input( "Enter your name: ") #prompt
age = input( "Enter your age: ")
print("Hello " + name + "! you are " + age)

#Calculadora
num1 = input("Enter a number: ") #es string por default
num2 = input("Enter another number: ")
result = float(num1) + float(num2) #tambien puedo usar int()
print(result)



# Mad Libs

color = input("Enter a color: ")
plural_noun = input("Enter a Plural Noun: ")
celebrity = input("Enter a celebrity: ")

print("Roses are " + color)
print(plural_noun + " are blue")
print(" I love " + celebrity)

# Lists

friends = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]
lucky_numbers = [4, 8, 15, 16, 23, 42] 
#can have different types of values
l2 = ["Kevin", 2, True]
# access the first element of the list
print(friends[0])
print(friends[-1]) #last element
print(friends[1:3]) #tomar un rango
print(friends[1:]) #todos a partir de Karen

#Modificar un elemento de la lista
friends[1] = "Dwight"
print(friends)


# List Functions

friends.extend(lucky_numbers) #add two lists together
friends.append("Creed")
friends.insert(1, "Kelly") #con un indice en especifico
friends.remove("Oscar")
print(friends.index("Creed")) #me da el indice en dónde está
print(friends.count("Jim")) #cuantas veces aparece
lucky_numbers.sort()

friends2 = friends.copy()
print(friends2)


# Tuples

#inmutable, no se puede cambiar
coordinates = (4, 5)
print(coordinates[0])


# Functions

# has to be indented
def say_hi(name, age):
    print("Hello " + name + ", you are " + age)

say_hi("Ace", "24")

#return statement: get information back from the function
def cube(num):
    return num*num*num

result = cube(4)
print(result)


# If Statements
is_male = False
is_tall = True

if is_male or is_tall:
    print("You are a male or tall or both")
else:
    print("You are neither male nor tall")    

if is_male and is_tall:
    print("You are a tall male ")
elif is_male and not(is_tall):
    print("You are a short male")
elif not(is_male) and is_tall:
    print("You are a tall woman")
else:
    print("You are either not male or not tall or both")

# Comparisons
def max_num(num1, num2, num3):
        if num1 >= num2 and num1 >= num3:
            return num1
        elif num2 >= num1 and num2 >= num3:
            return num2
        else:
            return num3


print(max_num(3, 11, 7))



# Diccionarios

# key: value
monthConversions = {
    "Jan": "January",
    "Feb": "Febuary",
    "Mar": "March",
    12 : "December"
}

print(monthConversions.get(12, "Not a valid key")) #el segundo es un default value, por si no existe Feb


# While Loop

i = 1
while i <= 10: #condición, mientras sea cierta va a seguir con lo de adentro
    print(i)
    i += 1 #es lo mismo que poner i=i+1


# For Loop

for letter in "Giraffe Academy":
    print (letter)

friends = ["Kevin", "Karen", "Jim", "Oscar", "Toby"]

for friend in friends:
    print(friend)

for index in range(10): #range da una sucesión de enteros (0 a 10 sin incluir el 10)
    print(index)

for index in range(4, 10): #4 a 9 (sí incluye el 4)
    print(index)

for index in range(len(friends)): 
    print(friends[index])

for i in range(5):
    if i == 0:
        print("This is the first iteration")
    else:
        print("Not first")

# Try/Except

try:
    number = int(input(("Enter a number: ")))
    print(number)
except:
    print("Invalid Input") #en lugar de breaking the program, agrupa todos los errores


# Reading Files

file = open("dat.csv", "r") # r: solo quiero leer
print(file.readable()) #me dice si lo puedo leer
print(file.readline()) #primera linea, readlines() para todas


file.close()




import os
os.getcwd()




# CLASES DE PYTHON




from math import sqrt, atan 

# pedir ayuda
?str

# asignaciones dobles
a, b = 3, 4
print("El valor de a es ", a)
print("El valor de b es ", b)

# tipos de variables

# entero
x = 3

# flotante
y = 3.5

# complejos
z = 3 + 5j
print(type(z))

# booleanos
a = True

# cadenas de texto (strings)
cadena = "Esto es una cadena con acentos y otros símbolos /&%$$#¨*..."
print(cadena.upper()) #volver a mayusculas)


# asignar el tipo de variable de forma explicita
x = int(3) 
y = float(3)
z = complex(1+2j)
a = str("Hola")
print(x, y, z, a)

# Operaciones
y1 = x**y #exponente
y2 = y/x #division
y3 = x%y # modulo
print(y1)

x += 1 # suma y asigna

# Operadores lógicos
x == 1
x != 1 #diferente
x > 10 or x <= 5
x > 10 and x <= 5

# Bucles

# tupla: conjunto de valores ordenados e inmutables
for i in (1,2,3,4,5):
    print(i)

for i in "Hola como estas":
    print(i)    

# while
i = 1
while i > 0 and i <= 5:
    print(i)
    i += 1

# for, range
for i in range(3, 100, 16): # inicio, final y tamaño de paso, no toma el ultimo valor
    print(i)

for i in range(1, 10): 
    print(i)

# Listas

a_list = [1, 2, 3, 4, 5]
print(a_list)
another_list = [1, 2, 3+5j , 2.345, 3, 5, 2.71] #no necesariamente son del mismo tipo
print(another_list)

print(another_list[2]) #python cuenta desde cero
print(another_list[-1]) #el ultimo

# List Slicing
print(another_list[1:]) #todos a partir del segundo
print(another_list[2:4])
print(another_list[:4]) #hasta el 4

#Append
n = 5
a_list.append(n) #agrega el 5 al final de la lista
print(a_list)
a_list.pop() #borra el ultimo elemento de la lista
print(a_list)

a_list[1] = 11
print(a_list)

# List Comprehensions
# describir a las listas como conjuntos

S = [x**2 for x in range(10)]
V = [2**x for x in range(13)]
M = [x for x in S if x%2 == 0] #los pares en S

print(S)
print(V)
print(M)


#suma loca
suma1 = sum([1/(i**2) for i in range(1,1001)])
print(suma1)

#lista de strings
palabras = 'Anita lava la tina'.split() #sin argumentos, hace el split con espacios
print(palabras)
transformacion = [[w.upper(), w.lower(), len(w)] for w in palabras] "array de arrays"
print(transformacion)


# Diccionarios

#un conjunto de parejas: llave ---> valor
diccionario = {'a':1, 'b':2, 'coral':'amarillo'}
diccionario['a'] #llave: 'a', valor: 1
diccionario['coral']


# Funciones

def polar(x,y):
    """
    x, y son un par ordenado
    """
    r = sqrt(x**2 + y**2)
    theta = atan(y/x)
    return r, theta

#pedir ayuda, creo que solo en jupyter
polar?
polar??
sum?
polar(1,1)

# NUMPY

# Arrays

#lista, puede tener varios tipos de datos
ls = [1, 2, 3, 4, 5]

#array, un solo tipo de datos, más velocidad
arr = np.array([1, 2, 3, 4, 5])

arr = np.linspace(0,1,100) #entre 0 y 1 genera 100 puntos (incluye al 0 y 1)
type(arr)

arr = np.logspace(0, 1, 100, base = 10) #espaciado? logaritmicamente

#array en 2 dimensiones
arr2d = np.zeros((5,5)) #matriz de ceros de 5x5

#identidad
np.eye(4)

# Reshaping

arr = np.arange(1000)
arr3d = arr.reshape((10,10,10))
arr3d.ndim

# Broadcasting: operaciones a todo el array

data = np.array([1, 2, 3, 4])
data + 1 # le suma uno a cada entrada
data**2


# Transponer
arr = np.arange(15).reshape((3,5)) #lo vuelve una matriz de 3x5
arr

#transpuesta:
arr_t = arr.T



# Slicing

arr = np.arange(10)
arr_slice = arr[5:8] # si modifico uno, modifico el otro
#ej.
arr_slice[2] = 4354.2 #tambien modifique al original, lo vuelve entero
arr_slice

#generar un nuevo array
arr2 = np.copy(arr)

# Multidimensional
arr = np.arange(9)
arr.shape = (3,3)
arr[2] #acceder el ultimo renglon
arr[1][1] #acceder renglon 2, col 2



