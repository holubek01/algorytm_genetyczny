import copy
import itertools
from dataclasses import dataclass
import random
import tsplib95
from itertools import permutations
import numpy

problem = tsplib95.load('C:\\ALL_tsp\\gr17.tsp\\gr17.tsp')

population_size = 10

#czy to wszystko potrzebne?
#tour = [0 for j in range(int(sizeTab))]
#optTour = [0 for j in range(int(sizeTab))]
zmienna = list(problem.get_nodes())
sizeTab = len(zmienna)
matr = [[0 for _ in range(sizeTab)] for _ in range(sizeTab)]

crossed_creatures = []
id_counter = 0

#klasa osobnik
@dataclass
class Creature:
    id: int
    genotyp: list
    fenotyp: int
    wiek: int
    wyspa: int


@dataclass
class Island:
    id: int
    creatures: list


tour1 = [0 for j in range(sizeTab)]

tour2 = [0 for j in range(sizeTab)]

tour3 = [0 for j in range(sizeTab)]

tour4 = [0 for j in range(sizeTab)]

tour5 = [0 for j in range(sizeTab)]


def crossing(tour1, tour2):
    for i in range(sizeTab):
        tour3[i] = i
    random.shuffle(tour3)

    j = tour3[random.randint(0,sizeTab-1)]
    k = tour3[random.randint(0,sizeTab-1)]
#swap
    if (k < j):
        m = j
        j = k
        k = m

    #przepisywanie elementów do pivota, za pivotem crossowanie
    for i in range(j, k + 1):
        tour4[i] = tour1[i]
        tour5[i] = tour2[i]

    helptour = [0 for _ in range(sizeTab - (k - j + 1))]
    helptour2 = [0 for _ in range(sizeTab - (k - j + 1))]
    index = 0

#przepisywanie do tablic pomocnicych genotyp bez odziedziczonych miast
    for i in range(0, sizeTab):
        if (i < j or i > k):
            helptour[index] = tour2[i]
            helptour2[index] = tour1[i]
            index = index + 1

    index = 0
    index2 = 0
    for i in range(0, sizeTab):
        for i2 in range(sizeTab - (k - j + 1)):
            #jesli element potarza sie to go przepisujemy (do 1 rodzica)
            if (tour1[i] == helptour[i2]):
                while (index >= j and index <= k):
                    index = index + 1
                tour5[index] = tour1[i]
                index = index + 1
                #do 2 rodzica analogicznie
            if (tour2[i] == helptour2[i2]):
                while (index2 >= j and index2 <= k):
                    index2 = index2 + 1
                tour4[index2] = tour2[i]
                index2 = index2 + 1

    global id_counter
    c1 = Creature(1, tour1, destination(sizeTab, matr, tour1), 4, 0)
    c2 = Creature(1, tour2, destination(sizeTab, matr, tour2), 4, 0)
    c4 = Creature(id_counter, tour4, destination(sizeTab, matr,tour4), 4, 0)
    c5 = Creature(id_counter+1, tour5, destination(sizeTab, matr,tour5), 4, 0)

    id_counter+=2
    print("Creature1 before crossing: ", c1.fenotyp," ", c1.genotyp)
    print("Creature2 before crossing: ", c2.fenotyp," ", c2.genotyp)
    print("Creature1 after crossing:  ", c4.fenotyp," ", c4.genotyp)
    print("Creature2 after crossing:  ", c5.fenotyp," ", c5.genotyp)
    print()

    global crossed_creatures
    crossed_creatures.append(copy.deepcopy(c4))
    crossed_creatures.append(copy.deepcopy(c5))

def destination(sizeTab, matr, tour3):
    weight = 0
    for i in range(0, int(sizeTab) - 1):
        weight += matr[tour3[i]][tour3[i + 1]]

    weight += matr[tour3[int(sizeTab) - 1]][tour3[0]]
    return weight


def getBest(list):
    #posortujemy liste po fentotypach
    list = sorted(list, key=lambda x: x.fenotyp, reverse=True)

    result = []
    result.append(list[0])
    result.append(list[1])

    print(list[0].id ," ", list[1].id)


    return result
#end getBest

    # best1 = list[0]
    # best2 = list[1]
    #
    # result = [0 for i in range(2)]
    #
    # fenotyp_list = []
    # for i in range(0, length):
    #     fenotyp_list.append(list[i])
    #
    # for i in range(1, length):
    #     if (list[i].fenotyp < best1.fenotyp):
    #         best1 = list[i]
    # fenotyp_list.remove(best1)
    # for i in range(0, length-1):
    #     if (fenotyp_list[i].fenotyp < best2.fenotyp):
            # best2 = fenotyp_list[i]
    #print(best1.fenotyp, " ", best2.fenotyp)

#    result[0] = best1
 #   result[1] = best2
  #  return result

def fill_matrix(sizeTab, matr, l):
    for i in range(0, sizeTab):
        for j in range(0, sizeTab):
            edge = i + l, j + l
            matr[i][j] = problem.get_weight(*edge)



#dlaczego tabu list jest tutaj nieuzywane ?
def getParentsToReproduction(island, tabulist):
    result = [0 for i in range(2)]
    firstId = random.randint(0, 9)
    secondId = random.randint(0, 9)
    while (firstId == secondId):
        secondId = random.randint(0, 9)

    print("First Id ", firstId, "Second Id ", secondId)
    result[0] = island.creatures[firstId]
    result[1] = island.creatures[secondId]

    return result


def main():
    k = problem.is_full_matrix()
    creatureslist = []
    tour_list = []

    if not k and not problem.is_explicit():
        fill_matrix(sizeTab, matr, 1)
    else:
        fill_matrix(sizeTab, matr, 0)

    #tworzymy pierwsza wyspe
    island = Island(0, creatureslist)
    for i in range(10):
        tour = tour1.copy()

        #wypelnij losowo tour
        for j in range(sizeTab):
            tour[j] = j
        random.shuffle(tour)

        tour_list.append(tour)

        c = Creature(i, tour, destination(sizeTab,matr,tour), 4, 0)
        island.creatures.append(c)


    tabulist = []
    for k in range(5):
        #parents = getBest(tournament(island))
        parents = roulette(island.creatures)

        while parents in tabulist:
            parents = roulette(island.creatures)
            #parents = getBest(tournament(island))

        tabulist.append(parents)

        los = random.randint(0,100)
        if los > 5:
            crossingPMX(parents[0].genotyp, parents[1].genotyp)

    global crossed_creatures
    for element in crossed_creatures:
        print(element)
    print('')
    print('')

    for element in crossed_creatures:
        losowa = random.randint(0,100)
        if losowa <= 5:
            print("ppb: ", losowa)
            print("id: ", element.id)
            #element.genotyp = invert(element.genotyp, random.randint(0,len(element.genotyp)-1), random.randint(0,len(element.genotyp)-1) )
            element.genotyp = heuristic_mutation(element.genotyp)
            element.fenotyp = destination(sizeTab, matr, element.genotyp)


    for element in crossed_creatures:
        print(element)




def invert(my_list, start, end):
    if start > end:
        temp = start
        start = end
        end = temp

    print("start: ", start)
    print("end: ", end)
    my_list[start:end] = my_list[start:end][::-1]
    return my_list


def tournament(givenIsland):
    list = []
    tabu = []

    for i in range(5):
        losowa = random.randint(0,9)
        while losowa in tabu:
            losowa = random.randint(0,9)

        #w tym miejsu juz na pewno losowej nie ma w tabu
        list.append(givenIsland.creatures[losowa])
        tabu.append(losowa)

    return list


def crossingPMX(parent1, parent2):
    j = random.randint(0, sizeTab-1)
    k = random.randint(0, sizeTab-1)

    child1 = [0 for j in range(sizeTab)]
    child2 = [0 for j in range(sizeTab)]
    if (k < j):
        m = j
        j = k
        k = m

    print("min ", j, "max ", k)
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
        while (index1 >= j and index1 <= k):
            index1 = index1 + 1
        while (index2 >= j and index2 <= k):
            index2 = index2 + 1
        for i2 in range(j, k + 1):
            if (helpparent1[i] == child1[i2]):
                repeater = i2
                value = mpxHelper(child1, parent1, repeater, j, k)
                child1[index1] = value
                go1 = 1
            if (helpparent2[i] == child2[i2]):
                repeater = i2
                value = mpxHelper(child2, parent2, repeater, j, k)
                child2[index2] = value
                go2 = 1

        if (go1 == 0):
            child1[index1] = helpparent1[i]
        else:
            go1 = 0
        if (go2 == 0):
            child2[index2] = helpparent2[i]
        else:
            go2 = 0
        index1 = index1 + 1
        index2 = index2 + 1

    print(helpparent1)
    print(helpparent2)
    print("Creature1 before crossing: ", parent1)
    print("Creature2 before crossing: ", parent2)
    print("Creature1 after crossing:  ", child1)
    print("Creature2 after crossing:  ", child2)
    print()

    global id_counter
    c4 = Creature(id_counter, child1, destination(sizeTab, matr, child1), 4, 0)
    c5 = Creature(id_counter + 1, child2, destination(sizeTab, matr, child2), 4, 0)
    id_counter += 2

    global crossed_creatures
    crossed_creatures.append(copy.deepcopy(c4))
    crossed_creatures.append(copy.deepcopy(c5))


def mpxHelper(child, parent, value, j, k):
    for i in range(j, k + 1):
        if (parent[value] == child[i]):
            return mpxHelper(child, parent, i, j, k)
    return parent[value]


#algorytm wybiera k losowych miast, tworzy ich permutacje, znajduje najlepsza z nich (najmniejsza
# wartosc funkcji celu po wstawieniu do genotypu) i nadpisuje genotyp osobnika
def heuristic_mutation(tour):
    maxValue = 4
    losowa = random.randint(2, maxValue)

    losowe_miasta = []
    losowe_miasta_indeksy = []


    for i in range(losowa):
        losowe_miasto = random.randint(0,len(tour)-1)

        while losowe_miasto in losowe_miasta_indeksy:
            losowe_miasto = random.randint(0, len(tour)-1)

        losowe_miasta.append(tour[losowe_miasto])
        losowe_miasta_indeksy.append(losowe_miasto)

    print("pocztkowy tour: ", tour)
    print("losowe miasta indeksy: ", losowe_miasta_indeksy)
    #wszystkie losowe permutacje
    permutacje = list(itertools.permutations(losowe_miasta))

    best_perm = losowe_miasta
    min = destination(sizeTab, matr, tour)

    for perm in permutacje:
        counter = 0
        for index in losowe_miasta_indeksy:
            tour[index] = perm[counter]
            counter+=1
        counter = 0

        #wstawilo nowe indeksy do toura (zamnienilo losowe miasta)
        #teraz policz dla nich destynacje i zobacz czy jest mniejsza od min
        if destination(sizeTab,matr, tour) < min:
            min = destination(sizeTab, matr, tour)
            best_perm = perm
            print(perm)
            print("min: ", min)
            print("tourrr", tour)
        #permutacja zapamietana - potem nalezy ja odtworzyc

    #odtworz najleosza permutacje
    counter = 0
    for index in losowe_miasta_indeksy:
        tour[index] = best_perm[counter]
        counter += 1
    counter = 0

    return tour



def crossingCX(parent1, parent2):
    j = random.randint(0,sizeTab-1)
    k = random.randint(0,sizeTab-1)

    child1 = [0 for j in range(sizeTab)]
    child2 = [0 for j in range(sizeTab)]
    help1 = []
    help2 = []

    start = parent1[0]
    end = parent2[0]
    pivot = 0

    helpList = []
    print(parent1)
    print(parent2)
    helpList.append(start)
    while(start != end):
        helpList.append(end)
        for i in range(0, sizeTab):
            if (parent1[i] == end):
                pivot = parent2[i]
        end = pivot

    for i in range(0, sizeTab):
        go1 = 0
        go2 = 0
        for i2 in helpList:
            if(parent1[i] == i2):
                go1 = 1
            if(parent2[i] == i2):
                go2 = 1
        if(go1 == 0):
            help1.append(parent1[i])
        else:
            go1 = 0
        if(go2 == 0):
            help2.append(parent2[i])
        else:
            go2 = 0

    index1 = 0
    index2 = 0
    go1 = 0
    go2 = 0
    for i in range(0, sizeTab):
        for i2 in helpList:
            if(parent1[i] == i2):
                child1[i] = i2
                go1 = 1


            if(parent2[i] == i2):
                child2[i] = i2
                go2 = 1

        if(go1 == 0):
            child1[i] = help2[index1]
            index1 = index1 + 1
        else:
            go1 = 0
        if(go2 == 0):
            child2[i] = help1[index2]
            index2 = index2 + 1
        else:
            go2 = 0
    print(child1)
    print(child2)

    global id_counter
    c4 = Creature(id_counter, child1, destination(sizeTab, matr, child1), 4, 0)
    c5 = Creature(id_counter + 1, child2, destination(sizeTab, matr, child2), 4, 0)
    id_counter+=2

    global crossed_creatures
    crossed_creatures.append(copy.deepcopy(c4))
    crossed_creatures.append(copy.deepcopy(c5))


def roulette(lista_osobnikow):
    p = []

    for i in range(int(0.1*population_size)):
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
        #losowa = numpy.random.choice(numpy.arange(0, population_size), p)
        while losowa[0] in tabu:
            losowa = random.choices(indeksy, p)
            #losowa = numpy.random.choice(numpy.arange(0, population_size), p)

        list.append(lista_osobnikow[losowa[0]])
        tabu.append(losowa)

    return list

    #mamy 2 losowych osobnikow ale wybranych z rozkładem jakims tam


if __name__ == "__main__":
    main()



#1 osobnik z 2opta

#elitaryzm ???

#najslabszego wrzucamy do neighboura

#pamietac o insercie do badań (wybor mutacji)

#"sortujemy po fenotypie i jesli np 20 bedzie mialo te same wartosci to wybieramy losowego z tych 20 i robimy
#2 inserty losowe na nim" - Piotr 2022

#warunek stopu to może być zarówno ilość pokoleń np 1000 albo ilosc pokolen bez poprawy !!!

#gotowe, smacznego !