x = [{"x": 1, "y": 8}, {"x": 2, "y": 2}, {"x": 3, "y": 3}]
y = {"x": 200, "y": 3}


def cost_function_median(x, y):
    suma = 0
    for i in x:
        print(i["x"], i["y"])
        suma += (i["x"] - y["x"])**2 + (i["y"] - y["y"])**2
    return suma


print(cost_function_median(x, y))