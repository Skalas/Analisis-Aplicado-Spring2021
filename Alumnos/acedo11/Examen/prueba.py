#prueba examen

l = []
for i in range(1000):
    if  i % 5 == 0 or i % 3 ==0:
        l.append(i)

print(sum(l))

