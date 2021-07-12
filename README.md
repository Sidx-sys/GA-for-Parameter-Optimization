## GA for Parameter Optimization

Our Genetic Algorithm had **3 main sections.** The sections were : 

- Generate Initial Population → Generated the inital population from a given seed
- Compute Fitness → Computed the fitness of all the indivuduals in a generation based on a fitness function
- Breed New Population → Calculates the likelihood of an individual to be chosen as a parent according to their fitness scores, *generates offspring using crossover,* and probabilistically mutates it
  
---
### Generating Initial Population

The first step in our Genetic Algorithm was to *create a population to start the process* of evolution from a given seed. We *initially took the overfit vector* provided to us in the project as the seed, but as we progressed in the project, we found better vectors to use as our seed so that we could achieve better results in further generations.

For creating the initial population, we **took the seed and added noise to each element** in that seed.

```python
seed = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
            8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
population = []
for _ in range(pop_size):
    individual = [w for w in seed]
    for i in range(len(individual)):
        individual[i] += noise[i]()
    population.append(individual)
return population
```

After some trial and error, we decided to choose our noise such that it only creates small changes to the elements of the seed vector to create an initial population to work with. For that,, we **chose a random number from the range** ⇒ `(-seed[i]/10, seed[i]/10)` and add this to `seed[i]` to create the initial population.

### Fitness Function

If there is something that we have changed numerous times throughout the making of our GA, it is the fitness function that we've used. The arguments to our fitness functions were the training and the validation mean squared errors of the given vector, that that we got from making0 API calls.

A lot of thought went into what the fitness function should be, keeping in mind how far apart on average the `training_mse` and `validation_mse` were for the current generation. Based on which one we chose to reduce or increase, we would modify the fitness function.

Initially, when we started out with the **overfit vector**, it had a high `validation_mse`(3e11) but a comparatively low `training_mse`(1e10). So here, our focus was to balance out the `validation_mse` and the `training_mse`. This could be achieved by making them meet at some middle ground, which would require that we incentivize reduction in the `validation_mse` and increase in the `training_mse`. As in the conventional GA, our GA's objective was to come up with vectors that maximize the output of the fitness function.

The first fitness function we used was $\frac{1}{0.8v+0.2t}$ where `v` is the `validation_mse` and `t` is `training_mse`. Our objective was to provide the GA more incentive to reduce the `validation_mse` than to reduce the `training_mse`. However, this fitness function didn't achieve the results we hoped for.

So we decided to take up a 2 step process.

The first step was intentionally make the GA want to reduce `training_mse` and increase the `validation_mse`. We achieved this by using $\frac{t}{v^2}$ as our fitness function. This function heavily rewards decrease in the `validation_mse` while penalizing decrease in `training_mse`. We ran this for a few generations and monitored the average fitness of each generation till the `training_mse` and `validation_mse` evened out. Once we achieved this balance, we could no longer rely on this fitness function as it would keep increasing the `training_mse`.

Onto the second step, since we had now balanced out the `training_mse` and `validation_mse`, simply reducing them both together seemed like the obvious next step. So we used $\frac{1}{xt+yv}$ as out fitness function, varying the values of $x$ and $y$ based on the values of `validation_mse` and `training_mse` that we got for the previous generation.

```python
train_mse = []
val_mse = []
for individual in population:
    err = api.get_errors(individual)
    train_mse.append(err[0])
    val_mse.append(err[1])
fitness = [t/v**2 for t, v in zip(train_mse, val_mse)]
return fitness, train_mse, val_mse
```

### Selection

Selection is how we decide the gene pool from our current generation for our next generation. A lot of different resources suggested a lot of different methods for selection. We decided to go with a probabilistic model based on the fitness of each individual in the generation.

```python
likelihood[i] = fitness[i]/sum(fitness)
```

Here, `fitness` is an array of the fitness values of all the individuals in a generation. `likelihood` is the probability distribution that we follow to pick a parent:

```python
parent1, parent2 = np.random.choice(population, likelihood, k=2)
```

So the individuals who are fitter have higher likelihood of getting selected and passing on their genes to the next generation. However, using this strategy, even the unfit ones have some chance, though very small, to be selected as one of the parents. This adds more variability to our generations, while still preserving the general trend that the fitter individuals pass on their genes to the next generation.

In our initial approach, we would create a completely new generation from our existing generation. But following some resources, we learnt that it is generally a good idea carry over the fittest parents to the next generation as well. So we create a `carry_over_factor` that determines what fraction of the current population gets carried over to the next generation. The fittest fraction of the population is carried over to the next population unchanged. This ensures that no even if the crossover or mutation turns up a bad bunch (due to their probabilistic nature), we would still be able to preserve at least the fitness levels of the previous generation.

### Crossover

We tried a couple of different crossover methods to use for our Genetic Algorithm. Initially, we would *choose two parents and take the element-wise average of them to create a new vector that was the offspring*. Clearly, this was very basic and from inspection of the results we realized that this crossover method was leading us to a dead-end.

Our second approach was to use a single split crossover, where we created a new offspring by taking the first half of the first parent and the second half of the second parent. This proved to be ineffective as a single vector would take over the initial population within just a few generations.

After some more research and study on crossover algorithms, we then used a method similar to **Davis' Order Crossover**, where we choose two parents and then split both of them at 2 randomly chosen points. Then we create two offsprings from them by *choosing the middle section of one parent and the side sections from the other parent.* While this algorithm was not bad, the results we're satisfactory enough.

Finally we landed on the **Binary Crossover** algorithm, since that added a good amount of variability in the crossover function.

The entire idea behind binary crossover is to generate two children from two parents, satisfying the following equation, all the while being able to control the variation between the parents and children using the distribution index value.

$$\frac{x_1^{new} + x_2^{new}}{2} = \frac{x_1 + x_2}{2}$$

The crossover is done by choosing a random number ($u$) in the range $[0, 1)$. The distribution index ($\eta_c$) is assigned a value between $[2, 5]$, it determines how much the offsprings will differ from their parents (*inversely related to difference*), and then $\beta$ is calculated as follows:

$$\beta = \begin{cases} 
        (2u)^{\frac{1}{\eta_c + 1}}, & \text{if } u \leq 0.5 \\
        {(\frac{1}{2(1-u)})}^{\frac{1}{\eta_c + 1}}, & \text{if } u \gt 0.5 \\
        \end{cases}$$

Then,

$$x_1^{new} = 0.5[(1+\beta)x_1 + (1-\beta)x_2] \\
x_2^{new} = 0.5[(1-\beta)x_1 + (1+\beta)x_2]$$

```python
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
```

For the next generation we *randomly choose one of the two offsprings* created.

The crossover is done by choosing a random number ($u$) in the range $[0, 1)$. The distribution index ($\eta_c$) is assigned a value between $[2, 5]$, it determines how much the offsprings will differ from their parents (*inversely related to difference*), and then $\beta$ is calculated as follows:

$$\beta = \begin{cases} 
        (2u)^{\frac{1}{\eta_c + 1}}, & \text{if } u \leq 0.5 \\
        {(\frac{1}{2(1-u)})}^{\frac{1}{\eta_c + 1}}, & \text{if } u \gt 0.5 \\
        \end{cases}$$

Then,

$$x_1^{new} = 0.5[(1+\beta)x_1 + (1-\beta)x_2] \\
x_2^{new} = 0.5[(1-\beta)x_1 + (1+\beta)x_2]$$

```python
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
```

For the next generation we *randomly choose one of the two offsprings* created.

### Heuristics

- **Having a good seed or initial vector helps the GA a lot.** It leads to finer vectors with much ease.
- The **Fitness Function** is really the most important factor when it comes to pointing your GA in the direction you want the generations to proceed. Although at times, it won't work exactly how one might think it will, we defined our fitness to be `t/v` once, thinking that it would tend to increase training and decrease the validation to improve fitness, but it ended up only increasing the `training_error` and did almost nothing for the `validation_error`.
- The amount of **noise** that needs to be added to the sample has to be carefully selected, as a noise too large for the sample destroys its meaning.