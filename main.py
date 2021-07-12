import numpy as np
from apicalls import API
import json
import random

api = API()

noise = [
    lambda : np.random.uniform(-1e-01, 1e-01),
    lambda : np.random.uniform(-1e-13, 1e-13),
    lambda : np.random.uniform(-1e-14, 1e-14),
    lambda : np.random.uniform(-1e-12, 1e-12),
    lambda : np.random.uniform(-1e-11, 1e-11),
    lambda : np.random.uniform(-1e-16, 1e-16),
    lambda : np.random.uniform(-1e-17, 1e-17),
    lambda : np.random.uniform(-1e-06, 1e-06),
    lambda : np.random.uniform(-1e-07, 1e-07),
    lambda : np.random.uniform(-1e-09, 1e-09),
    lambda : np.random.uniform(-1e-11, 1e-11),
]
pop_size = 15
gen_count = 4
mutation_rate = 0.05
carry_over_factor = 0.25

print(f"API Calls left: {api.get_usage()}")

def generate_inital_population():
    seed = [0.038726014930157066, -1.5053606547686412e-12, -2.3409032293228586e-13, 4.6254309174337466e-11, -1.67450952562246e-10, -1.903344083859845e-15, 8.491703685801306e-16, 2.388827285371692e-05, -2.130921881957366e-06, -1.499933383743236e-08, 9.898503072096021e-10]
    population = []
    for _ in range(pop_size):
        individual = [w for w in seed]
        for i in range(len(individual)):
            individual[i] += noise[i]()
        population.append(individual)
    return population

def fetch_initial_population():
    try:
        with open('last_population_data', 'r') as f:
            last_population_data = f.read()
        return json.loads(last_population_data)
    except FileNotFoundError:
        population = generate_inital_population()
        fitness,train_mse,val_mse = compute_fitness(population)
        return population,fitness,train_mse,val_mse

def compute_fitness(population):
    train_mse = []
    val_mse = []
    for individual in population:
        err = api.get_errors(individual)
        train_mse.append(err[0])
        val_mse.append(err[1])
    fitness = [1/(0.8*v + 0.2*t) for (t,v) in zip(train_mse,val_mse)]
    return fitness,train_mse,val_mse

def crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    u = random.uniform(0,1)
    nc = 3

    if u < 0.5:
        beta = (2*u)**(1/(nc + 1))
    else:
        beta = 1/((2*(1-u))**(1/(nc+1)))
    
    offspring1 = 0.5*((1+beta)*parent1  + (1-beta)*parent2)
    offspring2 = 0.5*((1-beta)*parent1 + (1+beta)*parent2)
    return random.choice([offspring1.tolist(),offspring2.tolist()])


def breed_new_population(population, fitness):
    new_population = []
    total_fitness = sum(fitness)
    likelihood = [(f/total_fitness) for f in fitness]
    
    sorted_fitness_indices = list(reversed(np.argsort(fitness)))
    parent_count = int(pop_size*carry_over_factor)

    for i in range(parent_count):
        new_population.append(population[sorted_fitness_indices[i]])
    
    for i in range(pop_size - parent_count):
        parent1,parent2 = random.choices(population, likelihood, k=2)
        offspring = crossover(parent1, parent2)

        if random.uniform(0,1) <= mutation_rate:
            for i in range(len(offspring)):
                if random.uniform(0,1) >= 0.75:
                    offspring[i] = 100*noise[i]()

        new_population.append(offspring)
    return new_population

def main():
    population,fitness,train_mse,val_mse = fetch_initial_population()
    print("-----------------------------Initial Population Fitness---------------------------")
    print(f"Best Fitness: {max(fitness)}")
    for i,f,t,v in zip(population,fitness,train_mse,val_mse):
        print(i, end=' => ')
        print(f,t,v)

    for gen_num in range(gen_count-1):
        new_population = breed_new_population(population, fitness)
        population = [individual for individual in new_population]
        fitness,train_mse,val_mse = compute_fitness(population)
        print(f"---------------------------------Generation {gen_num + 1}/{gen_count}------------------------")
        print(f"Best Fitness: {max(fitness)}")
        for i,f,t,v in zip(population,fitness,train_mse,val_mse):
            print(i, end=' => ')
            print(f,t,v)
    last_population_data = json.dumps([population,fitness,train_mse,val_mse])
    with open('last_population_data', 'w+') as f:
        f.write(last_population_data)

if __name__ == "__main__":
    main()