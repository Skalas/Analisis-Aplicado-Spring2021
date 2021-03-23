"""Implementar algoritmo de búsqueda lineal"""
### x0 definido
### while no se cumplen de paro()
    ### calcular pk
        ### para esto resolvemos el sistema lineal de la Hessiana*pk- gradiente (en caso de ser positiva def)
        ### en caso de no ser positiva definida, la hacemos positiva definida
    ### xk+1 = xk + alphak * pk
