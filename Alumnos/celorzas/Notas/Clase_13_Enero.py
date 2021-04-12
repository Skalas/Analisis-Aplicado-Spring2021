#import pandas as pandas


#x = [[1,1], [2,2], [3,3]] #serie de datos
x = [{"x": 1, "y": 8}, {"x": 2, "y": 2}, {"x": 3, "y": 3}] #jason --> característica " " : valor, llaves parte del proceso {} variable
y = {"x": 200, "y": 3}

def cost_function_median(x,y):
    suma = 0
    for i in x:
        print(i["x"], i["y"])
        suma += (i["x"] - y["x"])**2 + (i["y"] - y["y"])**2 #acceder a la caract. x sobre la variable i
    return suma

print(cost_function_median(x,y))


#def cost_function(x,y):
 #   return x + y 

#print("Hola, la función de costos es {},{}".format(cost_function(1,2), cost_function(2,4)))



