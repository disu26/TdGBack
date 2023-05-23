import random
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://localhost:8080/calculate",
    "https://trabajo-de-grado-f7c16.web.app"
]

app = FastAPI()

# Definición de los parámetros del algoritmo
W1 = 0.0
W2 = 0.0
W3 = 0.0

US_range = (0.0, 0.0)
UD_range = (0.0, 0.0)
DIO_range = (0.0, 0.0)
DSO_range = (0.0, 0.0)
DPO_range = (0.0, 0.0)
CF_range = (0.0, 0.0)
CI_range = (0.0, 0.0)
CT_range = (0.0, 0.0)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definición de la función de aptitud
def fitness_func(individual):
    UD = individual[0]
    US = individual[1]
    DIO = individual[2]
    DSO = individual[3]
    DPO = individual[4]
    CF = individual[5]
    CI = individual[6]
    CT = individual[7]

    fitness = W1*(UD/US) + W2*(DIO + DSO - DPO) - W3*(CF + CI + CT)
    return fitness

def generate_random_individual():
    individual = []
    individual.append(random.randint(UD_range[0], UD_range[1]))
    individual.append(random.randint(US_range[0], US_range[1]))
    individual.append(random.randint(DIO_range[0], DIO_range[1]))
    individual.append(random.randint(DSO_range[0], DSO_range[1]))
    individual.append(random.randint(DPO_range[0], DPO_range[1]))
    individual.append(random.uniform(CF_range[0], CF_range[1]))
    individual.append(random.uniform(CI_range[0], CI_range[1]))
    individual.append(random.uniform(CT_range[0], CT_range[1]))
    return individual
 
@app.post("/calculate")
async def calculate_values(w1: float, w2: float, w3: float, usMinRange: float, usMaxRange: float, udMinRange: float, udMaxRange: float, dioMinRange: float, dioMaxRange: float, dsoMinRange: float, dsoMaxRange: float, dpoMinRange: float, dpoMaxRange: float, cfMinRange: float, cfMaxRange: float, ciMinRange: float, ciMaxRange: float, ctMinRange: float, ctMaxRange: float):
    global W1
    global W2
    global W3

    global US_range
    global UD_range
    global DIO_range
    global DSO_range
    global DPO_range
    global CF_range
    global CI_range
    global CT_range

    W1 = w1
    W2 = w2
    W3 = w3

    US_range = (usMinRange, usMaxRange)
    UD_range = (udMinRange, udMaxRange)
    DIO_range = (dioMinRange, dioMaxRange)
    DSO_range = (dsoMinRange, dsoMaxRange)
    DPO_range = (dpoMinRange, dpoMaxRange)
    CF_range = (cfMinRange, cfMaxRange)
    CI_range = (ciMinRange, ciMaxRange)
    CT_range = (ctMinRange, ctMaxRange)

    population_size = 10
    population = [generate_random_individual() for _ in range(population_size)]

    # Definición del criterio de convergencia
    convergence_threshold = 10
    convergence_counter = 0
    best_fitness = -np.inf
    num_generations = 120

    for generation in range(num_generations):

        # Evaluación de la población
        fitness_scores = [fitness_func(individual) for individual in population]

        # Se seleccionan los 5 mejores individuos
        sorted_indices = np.argsort(fitness_scores)[::-1][:5]
        selected_population = [population[i] for i in sorted_indices]

        # Se realiza el cruce
        parent1, parent2 = random.choices(selected_population, k=2)
        crossover_point = random.randint(1, 7)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Se realiza la mutación
        # mutation_rate = 0.1
        # for i in range(5):
            # if random.random() < mutation_rate:
                # child[i] = random.uniform(0, 1)

        # Se evalua el nuevo individuo
        child_fitness = fitness_func(child)

        # Se reemplaza el peor individuo de la población original si el nuevo individuo es mejor
        worst_index = np.argmin(fitness_scores)
        if child_fitness > fitness_scores[worst_index]:
            population[worst_index] = child
            fitness_scores[worst_index] = child_fitness

        # Se actualiza el contador de convergencia
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            convergence_counter = 0
        else:
            convergence_counter += 1

        # Salir del bucle si se cumple el criterio de convergencia
        if convergence_counter >= convergence_threshold:
            break

    best_index = np.argmax(fitness_scores)

    return {"bestInd": population[best_index], "aptValue": abs(fitness_scores[best_index])}