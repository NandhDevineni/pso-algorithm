import random
import copy
import sys
import math
import csv


def sphere(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi*xi)
    # print("sphere function value " + fitnessVal);
    return fitnessVal


def cigar(position):
    fitnessVal = 0.0
    for i in range(1, len(position)):
        xi = position[i]
        fitnessVal += math.pow(10, 6) * (xi * xi)

    fitnessVal = (position[0] * position[0]) + fitnessVal
    # print("sphere function value " + fitnessVal);
    return fitnessVal


def discus(position):
    fitnessVal = 0.0
    for i in range(1, len(position)):
        xi = position[i]
        fitnessVal += (xi * xi)

    fitnessVal = ((position[0] * position[0]) * (math.pow(10, 6))) + fitnessVal
    # print("sphere function value " + fitnessVal);
    return fitnessVal


def rhe(position):
    fitnessVal = 0.0

    for i in range(len(position)):
        #xi = position[i]
        for j in range(i):
            xi = position[j]
            fitnessVal += xi * xi
    return fitnessVal


def zettl(position):
    fitnessVal = 0.0
    fitnessVal = ((1/4) * position[0]) + math.pow(
        ((position[0] * position[0]) - (2 * position[0]) + (position[1] * position[1])), 2)
    return fitnessVal


def leon(position):
    fitnessVal = 0.0
    sum1 = 0.0
    sum2 = 0.0

    sum1 = math.pow((position[1] - (position[0] * position[0])), 2)
    sum2 = math.pow((1 - position[0]), 2)

    fitnessVal = 100 * sum1 + sum2
    return fitnessVal


def easom(position):
    fitnessVal = 0.0

    temp1 = math.cos(position[0]) * math.cos(position[1])

    temp2 = ((position[0] - 3.14159) * (position[0] - 3.14159))
    temp3 = ((position[1] - 3.14159) * (position[1] - 3.14159))

    fitnessVal = - temp1 * math.exp(- temp2 - temp3)
    return fitnessVal


def zakharov(position):
    fitnessVal = 0.0
    temp = 0.0
    temp1 = 0.0

    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi*xi)
        temp += i * xi
        temp1 += temp * temp

    fitnessVal = fitnessVal + ((1/4) * (temp * temp)) + \
        ((1/16) * (temp1 * temp1))
    # print("sphere function value " + fitnessVal);
    return fitnessVal


def schwefel_1_2(position):
    fitnessVal = 0.0
    sum = 0.0

    for i in range(len(position)):
        #xi = position[i]
        for j in range(i):
            xi = position[j]
            sum += xi

        fitnessVal += sum * sum

    return fitnessVal


def rastrigin(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi))

    fitnessVal = 10 * 30 + fitnessVal
    return fitnessVal


def schwefel_2_26(position):
    fitnessVal = 0.0

    for i in range(len(position)):
        xi = position[i]
        fitnessVal += xi * math.sin(math.sqrt(abs(xi)))

    fitnessVal = (418.9829 * 30) - fitnessVal
    return fitnessVal


def michalewicz(position):
    fitnessVal = 0.0
    temp = 0.0
    temp1 = 0.0

    for i in range(len(position)):
        xi = position[i]
        temp = (i * xi * xi) / 3.14159
        temp_1 = math.pow(math.sin(temp), 20)

        fitnessVal += math.sin(xi) * temp_1
    return -(fitnessVal)


def styblinski_tang(position):
    fitnessVal = 0.0

    for i in range(len(position)):
        xi = position[i]

        fitnessVal += math.pow(xi, 4) - (16 * (xi * xi)) + 5 * xi

    fitnessVal = 1/2 * fitnessVal
    return fitnessVal


def schaffer_f_2(position):
    fitnessVal = 0.0
    temp1 = 0.0
    temp2 = 0.0

    x1 = position[0]
    x2 = position[1]

    temp1 = math.pow(math.sin((x1 * x1) - (x2 * x2)), 2) - 0.5
    temp2 = math.pow(1 + (0.001 * ((x1 * x1) + (x2 * x2))), 2)

    fitnessVal = 0.5 + temp1 / temp2
    return fitnessVal


def levy_13(position):

    fitnessVal = 0.0
    temp1 = 0.0
    temp2 = 0.0
    temp2 = 0.0
    x1 = position[0]
    x2 = position[1]

    temp1 = math.pow(math.sin(3 * 3.14159 * x1), 2)
    temp2 = math.pow((x1 - 1), 2) * \
        (1 + (math.pow(math.sin(3 * 3.14159 * x2), 2)))
    temp3 = math.pow((x2 - 1), 2) * \
        (1 + (math.pow(math.sin(2 * 3.14159 * x2), 2)))

    fitnessVal = temp1 + temp2 + temp3
    return fitnessVal


def ackley(position):
    fitnessVal = 0.0
    sum_1 = 0.0
    sum_2 = 0.0

    a = 20
    b = 0.2
    c = 2 * 3.14159

    for i in range(len(position)):
        xi = position[i]
        sum_1 += xi * xi
        sum_2 += math.cos(c * xi)

    fitnessval = (-1 * a) * (math.exp((-1*b)) * (math.sqrt((1/30)) * sum_1))

    fitnessVal = fitnessVal - (math.exp((1/30) * sum_2)) + a + math.exp(1)

    return fitnessVal


def rosenbrock(position):
    fitnessVal = 0.0
    temp1 = 0.0
    temp2 = 0.0

    for i in range(len(position)-1):
        xi = position[i]
        x2 = position[i+1]

        temp1 = math.pow(x2 - (xi * xi), 2)
        temp2 = math.pow((xi - 1), 2)

        fitnessVal += (100 * temp1) + temp2

    return fitnessVal


def sesw(position):
    fitnessVal = 0.0
    temp_1 = 0.0
    temp_2 = 0.0
    temp_3 = 0.0

    temp_1 = math.pow(math.sin(
        math.sqrt((position[0] * position[0]) + (position[1] * position[1]))), 2) - 0.5

    temp_2 = math.pow(
        1 + (0.001 * ((position[0] * position[0]) + (position[1] * position[1]))), 2)

    temp_3 = (temp_1 / temp_2) + 0.5

    for i in range(1, len(position)):
        fitnessVal += temp_3

    return fitnessVal


def schaffer_f_7(position):
    fitnessVal = 0.0
    temp_1 = 0.0
    temp_2 = 0.0

    for i in range(len(position)-1):
        xi = position[i]
        xi2 = position[i+1]

        yi = math.sqrt((xi * xi) + (xi2 * xi2))

        temp_1 = math.sin(50 * (math.pow(yi, 0.2)))
        temp_2 += math.sqrt(yi) + (temp_1 * math.sqrt(yi))

    fitnessVal = math.pow((29 * temp_2), 2)
    return fitnessVal

############################### Benchmark functions ended ################################################


class Particle:
    def __init__(self, fitness, dim, min, max, index):
        self.rnd = random.Random(index)

        self.position = [0.0 for i in range(dim)]

        self.velocity = [0.0 for i in range(dim)]

        self.p_best_pos = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = (min + (max - min) *
                                self.rnd.random())
            self.velocity[i] = (min + (max - min) *
                                self.rnd.random())

        # compute fitness of particle
        self.fitness = fitness(self.position)  # curr fitness

        # initialize best position and fitness of this particle
        self.p_best_pos = copy.copy(self.position)  # p best
        self.function_p_best_pos = self.fitness  # f(p_best)


def pso(fitness, max_iter, n, dim, min, max):
    #w = 0.65

    c1 = 2
    c2 = 2

    rnd = random.Random(0)

    # ****************** Chaotic Random Inertia Weight PSO ********************
    z = random.random()
    z = (4*z) * (1-z)
    #print("\n**********************************values of z is: ",z)
    w = (0.5*(random.random())) + (0.5*z)
    #################################################################

    population = [Particle(fitness, dim, min, max, i) for i in range(n)]

    gbest = [0.0 for i in range(dim)]
    f_gbest = sys.float_info.max  # population best

# calculating personal best and global best
    for i in range(n):
        if population[i].fitness < f_gbest:
            f_gbest = population[i].fitness  # f(g_best)
            gbest = copy.copy(population[i].position)  # g best

    # main loop of pso
    meanValue = 0.0
    sum = 0.0
    global_best_array = []

    Iter = 0
    fbest = []
    while Iter < max_iter:

        print("Iter = " + str(Iter) + " best fitness = %.3f" %
              f_gbest)
        fbest.append(f_gbest)

        for i in range(n):  # process each particle

            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()

                population[i].velocity[k] = (
                    (w * population[i].velocity[k]) +
                    (c1 * r1 * (population[i].p_best_pos[k] - population[i].position[k])) +
                    (c2 * r2 * (
                        gbest[k] - population[i].position[k]))
                )

                #print("\n**********************************values of w is: ",w)

                if population[i].velocity[k] < min:
                    population[i].velocity[k] = min
                elif population[i].velocity[k] > max:
                    population[i].velocity[k] = max

            # compute new position using new velocity
            for k in range(dim):
                population[i].position[k] += population[i].velocity[k]

            # compute fitness of new position
            population[i].fitness = fitness(population[i].position)

            # comparing previous p_best with current new position
            if population[i].fitness < population[i].function_p_best_pos:
                population[i].function_p_best_pos = population[i].fitness
                population[i].p_best_pos = copy.copy(population[i].position)

            # checking the if p_best is g_best
            if population[i].fitness < f_gbest:
                f_gbest = population[i].fitness
                gbest = copy.copy(population[i].position)

        # for-each particle
        Iter += 1
        # end_while
        global_best_array.append(f_gbest)
    # print("\n********************length of array is: ",len(global_best_array))

    for i in range(max_iter):
        meanValue += global_best_array[i]
    meanValue = meanValue / max_iter

    for i in range(len(global_best_array)):
        sum += ((global_best_array[i] - meanValue)
                * (global_best_array[i] - meanValue))

    std_dev = sum / max_iter
    std_dev = math.sqrt(std_dev)

    with open('mycsv.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['iteration', 'f_best'])
        i = 0
        for best in fbest:
            i = i + 1
            best = round(best, 3)
            thewriter.writerow([i, best])

    return gbest

# end pso


def switch_sphere():
    print("\n************************* Sphere function **************************\n")
    dim = 30
    fitness = sphere
    num_particles = 20
    max_iter = 1500

    print("Num_particles = " + str(num_particles))
    print("Max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)
    print("\n***************************End Sphere function*************************\n")


def switch_cigar():
    print("\n************************* CIGAR function **************************\n")
    dim = 30
    fitness = cigar
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -5.2, 5.2)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n***************************End CIGAR function*************************\n")


def switch_discus():
    print("\n************************* Discus function **************************\n")

    dim = 30
    fitness = discus
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n***************************End Discus function*************************\n")


def switch_rhe():
    print("\n************************* rhe function **************************\n")

    dim = 30
    fitness = rhe
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** End rhe function *************************\n")


def switch_zettl():
    print("\n************************* zettl function **************************\n")

    dim = 2
    fitness = zettl
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -5, 5)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** Enn zettl function *************************\n")


def switch_leon():
    print("\n************************* leon function **************************\n")

    dim = 2
    fitness = leon
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -1.2, 1.2)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** leon function *************************\n")


def switch_easom():
    print("\n************************* easom function **************************\n")

    dim = 2
    fitness = easom
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** easom function *************************\n")


def switch_zakharov():
    print("\n************************* zakharov function **************************\n")

    dim = 20
    fitness = zakharov

    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -5, 10)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n***************************End zakharov function*************************\n")


def switch_schwefel_1_2():
    print("\n************************* schwefel_1_2 function **************************\n")

    dim = 30
    fitness = schwefel_1_2
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** Enn schwefel_1_2 function *************************\n")


def switch_rastrigin():
    print("\n*************** Rastrigin function ********************\n")
    dim = 30
    fitness = rastrigin
    num_particles = 20
    max_iter = 1500

    print("Num_particles = " + str(num_particles))
    print("Max_iter    = " + str(max_iter))

    best_position = pso(fitness, max_iter, num_particles, dim, -5.2, 5.2)

    print("\nBest solution found:")
    print(["%.6f" % best_position[k] for k in range(dim)])
    fitnessVal = fitness(best_position)

    print("fitness of best solution = %.6f" % fitnessVal)

    print("\n***************************End rastrigin function*************************\n")


def switch_schwefel_2_26():
    print("\n************************* schwefel_2_26 function **************************\n")

    dim = 30

    fitness = schwefel_2_26
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -500, 500)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)

    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n***************************End schwefel_2_26 function*************************\n")


def switch_michalewicz():
    print("\n************************* michalewicz function **************************\n")

    dim = 2
    fitness = michalewicz
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, 0, 3.14159)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** michalewicz function *************************\n")


def switch_styblinski_tang():
    print("\n************************* styblinski_tang function **************************\n")

    dim = 30
    fitness = styblinski_tang
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -5, 5)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    fitnessVal = fitnessVal/dim
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** styblinski_tang function *************************\n")


def switch_schaffer_f_2():
    print("\n************************* schaffer_f_2 function **************************\n")

    dim = 2
    fitness = schaffer_f_2
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** schaffer_f_2 function *************************\n")


def switch_levy_13():
    print("\n************************* levy_13 function **************************\n")

    dim = 2
    fitness = levy_13
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -10, 10)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** levy_13 function *************************\n")


def switch_ackley():
    print("\n************************* ackley_function function **************************\n")

    dim = 30
    fitness = ackley
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -32, 32)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)
    print("\n*************************** End ackley function *************************\n")


def switch_rosenbrock():
    print("\n************************* rosenbrock function **************************\n")

    dim = 30
    fitness = rosenbrock
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -30, 30)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** rosenbrock function *************************\n")


def switch_sesw():
    print("\n************************* sesw function **************************\n")

    dim = 30
    fitness = sesw
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** Enn sesw function *************************\n")


def switch_schaffer_f_7():
    print("\n************************* schaffer_f_7 function **************************\n")

    dim = 30
    fitness = schaffer_f_7
    num_particles = 20
    max_iter = 1500

    print("Initializing num_particles = " + str(num_particles))
    print("Initializing max_iter = " + str(max_iter))

    optimal_position = pso(fitness, max_iter, num_particles, dim, -100, 100)

    print("\n Opitmal solution found: \n")
    print(["%.6f" % optimal_position[k] for k in range(dim)])

    fitnessVal = fitness(optimal_position)
    print("\n fitness of optimal solution = %.6f" % fitnessVal)

    print("\n*************************** Enn schaffer_f_7 function *************************\n")


choice = int(input("enter your choice: "))
operations = [switch_sphere,
              switch_cigar,
              switch_discus,
              switch_rhe,
              switch_zettl,
              switch_leon,
              switch_easom,
              switch_zakharov,
              switch_schwefel_1_2,
              switch_rastrigin,
              switch_schwefel_2_26,
              switch_michalewicz,
              switch_styblinski_tang,
              switch_schaffer_f_2,
              switch_levy_13,
              switch_ackley,
              switch_rosenbrock,
              switch_sesw,
              switch_schaffer_f_7]
output = operations[choice - 1]()
