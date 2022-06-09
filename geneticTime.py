import copy
import itertools
from dataclasses import dataclass
import random
import tsplib95
import numpy as np
import time

problem = tsplib95.load('C:\\Users\\holub\\OneDrive - Politechnika Wroclawska\\Desktop\\ALL_atsp\\att532.tsp\\brazil58.tsp')

population_size = 100
zmienna = list(problem.get_nodes())
sizeTab = 0
#matr = [[0 for _ in range(sizeTab)] for _ in range(sizeTab)]

matr = np.zeros((10,10))
crossed_creatures_list = [[] for _ in range(7)]
id_counter = 0
time1 = 0
time_counter = 0


@dataclass
class Creature:
    id: int
    genotyp: list
    fenotyp: int
    wiek: int
    wyspa: int
    ppb: float
    fitness: float


@dataclass
class Island:
    id: int
    creatures: list


tour1 = [0 for j in range(sizeTab)]
tour2 = [0 for j in range(sizeTab)]
tour3 = [0 for j in range(sizeTab)]
tour4 = [0 for j in range(sizeTab)]
tour5 = [0 for j in range(sizeTab)]


def crossing(tour1, tour2, id_wyspy):
    for i in range(sizeTab):
        tour3[i] = i
    random.shuffle(tour3)

    j = tour3[random.randint(0, sizeTab - 1)]
    k = tour3[random.randint(0, sizeTab - 1)]
    if (k < j):
        m = j
        j = k
        k = m

    for i in range(j, k + 1):
        tour4[i] = tour1[i]
        tour5[i] = tour2[i]

    helptour = [0 for _ in range(sizeTab - (k - j + 1))]
    helptour2 = [0 for _ in range(sizeTab - (k - j + 1))]
    index = 0

    for i in range(0, sizeTab):
        if (i < j or i > k):
            helptour[index] = tour2[i]
            helptour2[index] = tour1[i]
            index = index + 1

    index = 0
    index2 = 0
    for i in range(0, sizeTab):
        for i2 in range(sizeTab - (k - j + 1)):
            if tour1[i] == helptour[i2]:
                while j <= index <= k:
                    index = index + 1
                tour5[index] = tour1[i]
                index = index + 1
                # do 2 rodzica analogicznie
            if tour2[i] == helptour2[i2]:
                while index2 >= j and index2 <= k:
                    index2 = index2 + 1
                tour4[index2] = tour2[i]
                index2 = index2 + 1

    global id_counter
    c1 = Creature(1, tour1, destination(sizeTab, matr, tour1), 0, 0, 0, 1 / (destination(sizeTab, matr, tour1)))
    c2 = Creature(1, tour2, destination(sizeTab, matr, tour2), 0, 0, 0, 1 / (destination(sizeTab, matr, tour2)))
    c4 = Creature(id_counter, tour4, destination(sizeTab, matr, tour4), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, tour4)))
    c5 = Creature(id_counter + 1, tour5, destination(sizeTab, matr, tour5), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, tour5)))

    id_counter += 2

    global crossed_creatures_list
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c4))
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c5))


def destination(sizeTab, matr, tour3):
    weight = 0
    for i in range(0, int(sizeTab) - 1):
        weight += matr[tour3[i]][tour3[i + 1]]

    weight += matr[tour3[int(sizeTab) - 1]][tour3[0]]
    return weight


def getBest(list):
    #sortujemy liste po fentotypach
    list = sorted(list, key=lambda x: x.fenotyp)
    result = []
    result.append(list[0])
    result.append(list[1])
    return result


def getParentsToReproduction(island):
    result = [0 for i in range(2)]
    firstId = random.randint(0, 9)
    secondId = random.randint(0, 9)
    while firstId == secondId:
        secondId = random.randint(0, 9)

    result[0] = island.creatures[firstId]
    result[1] = island.creatures[secondId]

    return result


def create_and_fill_matrix(n):
    global matr
    matr = np.zeros((n, n))
    for i in range(n):
         for j in range(n):
            if i != j:
                 matr[i][j] = random.randint(0,1000)
    print(matr)


def main():
    global sizeTab
    while sizeTab <= 0:
        sizeTab = int(input("Podaj wielkość instancji: "))
        if sizeTab <= 0:
            print("Wielkosc instancji musi byc liczba dodatnia")

    create_and_fill_matrix(sizeTab)
    global tour1
    tour1 = [0 for j in range(sizeTab)]
    global tour2
    tour2 = [0 for j in range(sizeTab)]
    global tour3
    tour3 = [0 for j in range(sizeTab)]
    global tour4
    tour4 = [0 for j in range(sizeTab)]
    global tour5
    tour5 = [0 for j in range(sizeTab)]

    creatureslist = []
    island = Island(0, creatureslist)
    island2 = Island(1, creatureslist)
    island3 = Island(2, creatureslist)
    #island4 = Island(3, creatureslist)
    #island5 = Island(4, creatureslist)
    #island6 = Island(5, creatureslist)
    #island7 = Island(6, creatureslist)

    list_of_island = []
    list_of_island.append(island)
    list_of_island.append(island2)
    list_of_island.append(island3)
    #list_of_island.append(island4)
    #list_of_island.append(island5)
    #list_of_island.append(island6)
    #list_of_island.append(island7)

    connection_matrix = [[0 for _ in range(len(list_of_island))] for _ in range(len(list_of_island))]
    for i in range(len(list_of_island) - 1):
        connection_matrix[i][i + 1] = 1
        connection_matrix[i + 1][i] = 1

    connection_matrix[len(list_of_island) - 1][0] = 1
    connection_matrix[0][len(list_of_island) - 1] = 1

    # connection_matrix[0][2] = 1
    # connection_matrix[2][0] = 1
    # connection_matrix[1][4] = 1
    # connection_matrix[4][1] = 1
    #connection_matrix[3][6] = 1
    #connection_matrix[6][3] = 1

    t1 = time.time()
    stop_counter = 0
    for isl in list_of_island:
        create_first_population(isl)
    best = list_of_island[0].creatures[0].fenotyp
    while (stop_counter < 330 * len(list_of_island)):
        for isl in list_of_island:
            tab = make_children(isl, stop_counter, best)
            stop_counter = tab[1]
            best = tab[0]
        for k in range(3):
            for j in range(3):
                if connection_matrix[k][j] == 1:
                    losowa = random.randint(0, 100)
                    if losowa < 5:
                        losowa2 = random.randint(0, population_size - 1)
                        helper = list_of_island[k].creatures[losowa2]
                        list_of_island[k].creatures[losowa2] = list_of_island[j].creatures[losowa2]
                        list_of_island[j].creatures[losowa2] = helper
        print(stop_counter)
        print(best)
    print("best: ", best)

    t2 = time.time()

    global time1
    global time_counter
    print("time: ", time1/time_counter)
    print("time all: ", (t2-t1)*1000)


def make_children(island, stop_counter, best1):
    # island.creatures = generation(island)
    # best = island.creatures[0].fenotyp
    t = [0 for j in range(2)]
    iteration_counter = 0
    while iteration_counter < 100:
        a1 = time.time()
        island.creatures = generation(island)
        b1 = time.time()
        global time1
        time1 += ((b1-a1)*1000)
        global time_counter
        time_counter += 1
        stop_counter += 1
        iteration_counter += 1
        if best1 > island.creatures[0].fenotyp:
            best1 = island.creatures[0].fenotyp
            stop_counter = 0

    t[0] = best1
    t[1] = stop_counter
    return t


def create_first_population(island):
    for i in range(population_size - 1):
        tour = tour1.copy()
        for j in range(sizeTab):
            tour[j] = j
        random.shuffle(tour)

        c = Creature(i, tour, destination(sizeTab, matr, tour), 0, island.id, 0, 1 / (destination(sizeTab, matr, tour)))
        island.creatures.append(c)

    my_tour = tour.copy()
    optTour = [0 for j in range(int(sizeTab))]
    optTour[0] = random.randint(0, sizeTab - 1)
    my_tour.remove(optTour[0])
    last_child = close_neighbour(optTour, matr, my_tour)
    island.creatures.append(
        Creature(population_size - 1, last_child, destination(sizeTab, matr, last_child), 0, island.id, 0, 1 / (destination(sizeTab, matr, last_child))))


def generation(island):
    tabulist = []
    for k in range(int(0.4 * population_size)):
        # parents = getBest(tournament(island))
        parents = roulette(island.creatures)
        # parents = original_roulette(island.creatures)

        while parents in tabulist:
            parents = roulette(island.creatures)
            # parents = getBest(tournament(island))
            # parents = original_roulette(island.creatures)

        tabulist.append(parents)
        los = random.randint(0, 100)
        if los > 5:
            crossingPMX(parents[0].genotyp, parents[1].genotyp, island.id)
    sorted_by_fenotype = sorted(crossed_creatures_list[island.id], key=lambda x: x.fenotyp, reverse=True)
    worst_creature = sorted_by_fenotype[0]

    my_tour = worst_creature.genotyp.copy()
    optTour = [0 for j in range(int(sizeTab))]
    optTour[0] = random.randint(0, sizeTab - 1)
    my_tour.remove(optTour[0])
    my_tour2 = my_tour.copy()
    # zmienna2 = close_neighbour(optTour, matr, my_tour)

    if destination(sizeTab, matr, close_neighbour(optTour, matr, my_tour)) < worst_creature.fenotyp:
        worst_creature.genotyp = close_neighbour(optTour, matr, my_tour2)
        worst_creature.fenotyp = destination(sizeTab, matr, worst_creature.genotyp)
        worst_creature.fitness = 1 / worst_creature.fenotyp

    for element in crossed_creatures_list[island.id]:
        losowa = random.randint(0, 100)
        if losowa <= 5:
            # element.genotyp = invert(element.genotyp, random.randint(0,len(element.genotyp)-1), random.randint(0,len(element.genotyp)-1) )
            element.genotyp = heuristic_mutation(element.genotyp)
            element.fenotyp = destination(sizeTab, matr, element.genotyp)
            element.fitness = 1 / element.fitness
    sorted_by_fenotype = sorted(crossed_creatures_list[island.id], key=lambda x: x.fenotyp)

    counter = 0
    current = sorted_by_fenotype[0]
    for element in sorted_by_fenotype:
        if element.fenotyp == current.fenotyp:
            counter += 1
        else:
            counter = 0
            current = element

        if counter == 0.2 * population_size:
            invert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            invert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            invert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            # insert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            # insert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            # insert(current.genotyp, random.randint(0, sizeTab - 1), random.randint(0, sizeTab - 1))
            # current.genotyp = heuristic_mutation(current.genotyp)
            # current.genotyp = heuristic_mutation(current.genotyp)
            # current.genotyp = heuristic_mutation(current.genotyp)
            current.fenotyp = destination(sizeTab, matr, element.genotyp)
            current.fitness = 1 / current.fenotyp
            counter -= 1
            current = element

    sorted_island_creatures = sorted(island.creatures, key=lambda x: x.fenotyp)

    counter2 = 0
    while len(crossed_creatures_list[island.id]) < 1.2 * population_size:
        if sorted_island_creatures[counter2].wiek < 100:
            crossed_creatures_list[island.id].append(sorted_island_creatures[counter2])
            sorted_island_creatures[counter2].wiek += 1
        counter2 += 1

    crossed_creatures_list[island.id] = sorted(crossed_creatures_list[island.id], key=lambda x: x.fenotyp)

    while len(crossed_creatures_list[island.id]) > population_size:
        value = random.randint(20, len(crossed_creatures_list[island.id]) - 1)
        crossed_creatures_list[island.id].remove(crossed_creatures_list[island.id][value])
    return crossed_creatures_list[island.id]


def invert(my_list, start, end):
    if start > end:
        temp = start
        start = end
        end = temp

    my_list[start:end] = my_list[start:end][::-1]
    return my_list


def insert(my_list, start, end):
    k = my_list[start]
    my_list[start] = my_list[end]
    my_list[end] = k
    return my_list


def tournament(givenIsland):
    list = []
    tabu = []

    for i in range(5):
        losowa = random.randint(0, population_size - 1)
        while losowa in tabu:
            losowa = random.randint(0, population_size - 1)

        list.append(givenIsland.creatures[losowa])
        tabu.append(losowa)

    return list


def crossingPMX(parent1, parent2, id_wyspy):
    j = random.randint(0, sizeTab - 1)
    k = random.randint(0, sizeTab - 1)

    child1 = [0 for j in range(sizeTab)]
    child2 = [0 for j in range(sizeTab)]
    if (k < j):
        m = j
        j = k
        k = m

    for i in range(j, k + 1):
        child1[i] = parent2[i]
        child2[i] = parent1[i]

    helpparent1 = [0 for _ in range(sizeTab - (k - j + 1))]
    helpparent2 = [0 for _ in range(sizeTab - (k - j + 1))]
    helpchild1 = [0 for _ in range(sizeTab - (k - j + 1))]
    helpchild2 = [0 for _ in range(sizeTab - (k - j + 1))]

    index = 0
    for i in range(0, sizeTab):
        if (i < j or i > k):
            helpparent1[index] = parent1[i]
            helpparent2[index] = parent2[i]
            helpchild1[index] = child1[i]
            helpchild2[index] = child2[i]
            index = index + 1

    index1 = 0
    index2 = 0

    for i in range(sizeTab - (k - j + 1)):
        go1 = 0
        go2 = 0
        while j <= index1 <= k:
            index1 = index1 + 1
        while j <= index2 <= k:
            index2 = index2 + 1
        for i2 in range(j, k + 1):
            if helpparent1[i] == child1[i2]:
                repeater = i2
                value = mpxHelper(child1, parent1, repeater, j, k)
                child1[index1] = value
                go1 = 1
            if helpparent2[i] == child2[i2]:
                repeater = i2
                value = mpxHelper(child2, parent2, repeater, j, k)
                child2[index2] = value
                go2 = 1

        if go1 == 0:
            child1[index1] = helpparent1[i]
        else:
            go1 = 0
        if go2 == 0:
            child2[index2] = helpparent2[i]
        else:
            go2 = 0
        index1 = index1 + 1
        index2 = index2 + 1

    global id_counter
    c4 = Creature(id_counter, child1, destination(sizeTab, matr, child1), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, child1)))
    c5 = Creature(id_counter + 1, child2, destination(sizeTab, matr, child2), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, child2)))
    id_counter += 2

    global crossed_creatures_list
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c4))
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c5))


def mpxHelper(child, parent, value, j, k):
    for i in range(j, k + 1):
        if parent[value] == child[i]:
            return mpxHelper(child, parent, i, j, k)
    return parent[value]


# algorytm wybiera k losowych miast, tworzy ich permutacje, znajduje najlepsza z nich (najmniejsza
# wartosc funkcji celu po wstawieniu do genotypu) i nadpisuje genotyp osobnika
def heuristic_mutation(tour):
    maxValue = 4
    losowa = random.randint(4, maxValue)

    losowe_miasta = []
    losowe_miasta_indeksy = []

    for i in range(losowa):
        losowe_miasto = random.randint(0, len(tour) - 1)

        while losowe_miasto in losowe_miasta_indeksy:
            losowe_miasto = random.randint(0, len(tour) - 1)

        losowe_miasta.append(tour[losowe_miasto])
        losowe_miasta_indeksy.append(losowe_miasto)

    permutacje = list(itertools.permutations(losowe_miasta))

    best_perm = losowe_miasta
    min = destination(sizeTab, matr, tour)

    for perm in permutacje:
        counter = 0
        for index in losowe_miasta_indeksy:
            tour[index] = perm[counter]
            counter += 1

        # wstawilo nowe indeksy do toura (zamnienilo losowe miasta)
        # teraz policz dla nich destynacje i zobacz czy jest mniejsza od min
        if destination(sizeTab, matr, tour) < min:
            min = destination(sizeTab, matr, tour)
            best_perm = perm

    counter = 0
    for index in losowe_miasta_indeksy:
        tour[index] = best_perm[counter]
        counter += 1
    return tour


def crossingCX(parent1, parent2, id_wyspy):
    j = random.randint(0, sizeTab - 1)
    k = random.randint(0, sizeTab - 1)

    child1 = [0 for j in range(sizeTab)]
    child2 = [0 for j in range(sizeTab)]
    help1 = []
    help2 = []

    start = parent1[0]
    end = parent2[0]
    pivot = 0

    helpList = []
    helpList.append(start)
    while start != end:
        helpList.append(end)
        for i in range(0, sizeTab):
            if parent1[i] == end:
                pivot = parent2[i]
        end = pivot

    for i in range(0, sizeTab):
        go1 = 0
        go2 = 0
        for i2 in helpList:
            if parent1[i] == i2:
                go1 = 1
            if parent2[i] == i2:
                go2 = 1
        if go1 == 0:
            help1.append(parent1[i])
        else:
            go1 = 0
        if go2 == 0:
            help2.append(parent2[i])
        else:
            go2 = 0

    index1 = 0
    index2 = 0
    go1 = 0
    go2 = 0
    for i in range(0, sizeTab):
        for i2 in helpList:
            if parent1[i] == i2:
                child1[i] = i2
                go1 = 1

            if parent2[i] == i2:
                child2[i] = i2
                go2 = 1

        if go1 == 0:
            child1[i] = help2[index1]
            index1 = index1 + 1
        else:
            go1 = 0
        if go2 == 0:
            child2[i] = help1[index2]
            index2 = index2 + 1
        else:
            go2 = 0

    global id_counter
    c4 = Creature(id_counter, child1, destination(sizeTab, matr, child1), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, child1)))
    c5 = Creature(id_counter + 1, child2, destination(sizeTab, matr, child2), 0, id_wyspy, 0, 1 / (destination(sizeTab, matr, child2)))
    id_counter += 2

    global crossed_creatures_list
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c4))
    crossed_creatures_list[id_wyspy].append(copy.deepcopy(c5))


def close_neighbour(optTour, matr, tour):
    for i in range(0, int(sizeTab) - 1):
        cls = matr[optTour[i]][tour[0]]
        optTour[i + 1] = tour[0]
        if len(tour) > 1:
            for j in range(1, len(tour)):
                if matr[optTour[i]][tour[j]] < cls:
                    cls = matr[optTour[i]][tour[j]]
                    optTour[i + 1] = tour[j]
            tour.remove(optTour[i + 1])

        else:
            optTour[i + 1] = tour[0]
    return optTour


def original_roulette(lista_osobnikow):
    sum = 0
    sum_of_ppb = 0
    p = [0 for j in range(len(lista_osobnikow))]

    for member in lista_osobnikow:
        sum += 1 / member.fenotyp
    for j in range(len(lista_osobnikow)):
        p[j] = sum_of_ppb + ((1 / lista_osobnikow[j].fenotyp) / sum)
        sum_of_ppb += (1 / lista_osobnikow[j].fenotyp) / sum

    list = []
    mytabu = []
    for i in range(2):
        h = lista_osobnikow[len(lista_osobnikow) - 1]
        losowa = random.uniform(0, 1)
        for j in range(len(lista_osobnikow) - 1):
            if (losowa >= p[j] and losowa <= p[j + 1]):
                h = lista_osobnikow[j]
        while h in mytabu:
            losowa = random.uniform(0, 1)
            for j in range(len(lista_osobnikow) - 1):
                if (losowa >= p[j] and losowa <= p[j + 1]):
                    h = lista_osobnikow[j]

        list.append(h)
        mytabu.append(h)
    return list


def roulette(lista_osobnikow):
    p = []

    for i in range(int(0.1 * population_size)):
        p.append(0.18)

    for i in range(int(0.3 * population_size)):
        p.append(0.15)

    for i in range(int(0.2 * population_size)):
        p.append(0.1)

    for i in range(int(0.3 * population_size)):
        p.append(0.05)

    for i in range(int(0.1 * population_size)):
        p.append(0.02)

    lista_osobnikow = sorted(lista_osobnikow, key=lambda x: x.fenotyp)
    indeksy = []

    for i in range(population_size):
        indeksy.append(i)

    list = []
    tabu = []

    for i in range(2):
        losowa = random.choices(indeksy, p)
        while losowa[0] in tabu:
            losowa = random.choices(indeksy, p)

        list.append(lista_osobnikow[losowa[0]])
        tabu.append(losowa[0])

    return list


if __name__ == "__main__":
    main()

