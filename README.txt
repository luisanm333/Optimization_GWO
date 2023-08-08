Parámetros definidos por el usuario:

Max_corrida :    Número de veces que se ejecutará el algoritmo
Max_iter:        Criterio de parada (máximo número de iteraciones)
SearchAgents_no: Tamaño de la población o número de partículas 
LimInf:          Limite inferior del espacio de búsqueda
LimSup:          Limite superior del espacio de búsqueda

Parámetro definido por el problema
dim:          Número de características de la partícula (incógnitas del problema)


Los resultados de la optimización se almacenan en la Matriz 'Table_opt'
Donde: 
La primera columna es la corrida
Las n columnas siguientes son la ubicación de las partículas (solución al problema)
La última columna es el costo de la función

| corrida | x1 | x2 | ... | x Car | costo |

 
La curva de convergencia se almacena en el arreglo 'Table_Convergence_curve'.
Cada columna es la curva de convergencia de cada corrida

| corrida 1 | corrida 2 | ... | corrida Max_Corrida |
