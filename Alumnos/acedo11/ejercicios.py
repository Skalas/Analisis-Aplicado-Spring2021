
#    1  
#If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
#Find the sum of all the multiples of 3 or 5 below 1000.

l = []
for i in range(1000):
    if  i % 5 == 0 or i % 3 ==0:
        l.append(i)

print(sum(l))
# 233,168 (bien)


#    2

#By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.
def Fib(tope):
    l=[1, 2]
    i=2
    num = l[i-1] + l[i-2]
    while num <= tope:
        l.append(num)
        i += 1
        num = l[i-1] + l[i-2]
    return l

lista_fib = Fib(40000000)
lista_pares  = [x for x in lista_fib if x%2 == 0] #los pares en lista_fib
print(sum(lista_pares))
# 19,544,084 (mal)
#????


#   3

# What is the largest prime factor of the number 600851475143

def factores_primos(num):
    fact = []
    for i in range(1, num):
        if num % i ==0:
            fact.append(i)
    return fact

factores_resp = factores_primos(600851475143)
print(factores_resp[:10])



