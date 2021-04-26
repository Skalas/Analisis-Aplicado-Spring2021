% Propuesta de proyecto
% Equipo JAR (Julieta, José, Alonso, Rebe)

# Optimizando horas de estudio para maximizar la calificación final
## O: ¿Por fin voy a pasar eco 2?

# Planteamiento

## Función de calificación

Para no caer en un problema lineal consideramos funciones de
calificación no-lineales.

. . .

Por ejemplo: Peso de parciales depende de si es el más alto o bajo

$$
\frac{1}{5} p_{\text{lo}} + \frac{2}{5} \left( \sum_{i} p_{i} \right) + \frac{2}{5} y
$$

donde $y$ representa el examen final, $p_{\text{lo}}$ el parcial más
bajo, y así.

----

![Calificación final en función de parciales y final](figs/cs_simple_xkcd.png){.stretch}

# ¿Qué tan probable es tener ... Final?

----

Asumiendo que las calificaciones se distribuyen gamma

![Distribución normal con curvas de nivel en el piso](figs/normal_3d.png){.stretch}

## Juntando ambas ideas

![](figs/cs_compuesto_xkcd.png){.stretch}

----

![](figs/probab_xkcd.png){.stretch}
