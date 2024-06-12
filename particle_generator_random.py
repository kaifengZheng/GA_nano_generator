#!/home/kaifeng/miniconda3/envs/work/bin/python python3
# gauss
# making lattice
import sys

sys.path.append(".")
from tqdm.auto import tqdm
from itertools import *
import seaborn as sea
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.visualize import view
from random import sample, seed
from .shape_proj_util.geo_tools.geometry import *
from .shape_proj_util.geo_tools.lattice_maker import fccbasis, extendfcc
from ase.io.xyz import write_xyz
from scipy.spatial.distance import cdist
import random

# gauss
# making lattice
from tqdm.auto import tqdm


def empty_lattice(lc, n1, n2, n3):
    """
    Generate a fcc lattice by given lattice constant and superlattice parameters: n1,n2,n3.
    The constructed lattice is centered at (0,0,0), and the coordinates are reordered by the
    atom to center distance.
    @param lc: lattice constant
    @param n1: superlattice parameter: number of cells on x direction
    @param n2: superlattice parameter: number of cells on y direction
    @param n3: superlattice parameter: number of cells on z direction
    @param mid_position: the position of the center of the lattice, if None, the center of the lattice is set to (0,0,0)
    return: ordered lattice, mid_position
    """

    # lattice=extendfcc(fccbasis(lc),lc,n1,n2,n3)
    lattice = extendfcc(fccbasis(lc), lc, n1, n2, n3)
    mid_position = np.mean(lattice, axis=0)
    # lattice=np.round(lattice-mid_position,6)
    closest_index = np.where(
        np.all(np.isclose(mid_position, lattice, rtol=lc / (2 * np.sqrt(2))), axis=1)
    )[0][0]
    lattice = lattice - lattice[closest_index]
    return order_pos(lattice)


def order_pos(lattice, mid_position=None):
    if mid_position is None:
        mid_position = np.mean(lattice, axis=0)
    # reorder the lattice by distance to the center. The first atom is the closest to the center.
    dist = np.round(cdist([mid_position], lattice), 6)
    dist_dict = Counter(dist[0])
    sorted_count = sorted(dist_dict.items(), key=lambda x: x[0], reverse=False)
    record = []
    ordered_particle = []
    for i in range(len(sorted_count)):
        record.extend(np.where(dist == sorted_count[i][0])[1])
    for i in range(len(lattice)):
        ordered_particle.append(lattice[record[i]])
    return np.array(ordered_particle)


def particles_encode_gen_no_oblate(lc, num_atoms=None, max_num_atoms=None):
    if num_atoms == None:
        assert (
            max_num_atoms != None
        ), "Please provide the maximum number of atoms in the lattice."
        num_atoms = random.randint(12, max_num_atoms)
    # min_num_points=num_atoms
    num_lattice = max_num_atoms // 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    num_points = 0
    lattice_big = empty_lattice(lc, n_max, n_max, n_max)
    # transfer points to the big lattice
    indices = []
    genome = np.zeros(len(lattice_big))
    for i in range(num_atoms):
        indices.extend(
            np.where(
                np.all(np.isclose(lattice_big, lattice_big[i], rtol=1e-3), axis=1)
            )[0]
        )
    for i in indices:
        genome[i] = 1
    return genome, (n_max, n_max, n_max, num_atoms)


# generate random lattice with random n1,n2,n3 of the superlattice parimeters, and random numbers of atoms in the lattice.
def particles_encode_gen(lc, num_atoms=None, max_num_atoms=None):
    if num_atoms == None:
        assert (
            max_num_atoms != None
        ), "Please provide the maximum number of atoms in the lattice."
        num_atoms = random.randint(12, max_num_atoms)
    # min_num_points=num_atoms
    num_lattice = max_num_atoms // 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    num_points = 0
    lattice_big = empty_lattice(lc, n_max, n_max, n_max)
    # print(f"length={np.mean(lattice_big,axis=0)}")
    times = random.randint(1, 3)

    # this is too strong
    while num_points < num_atoms * 2:
        n1 = random.randint(1, n_max)
        n2 = n1
        n3 = random.randint(1, n1)

        lattice = empty_lattice(lc, n1, n2, n3)
        num_points = len(lattice)
    # print(f"length={np.mean(lattice,axis=0)}")
    # transfer points to the big lattice
    indices = []
    genome = np.zeros(len(lattice_big))
    for i in range(num_atoms):
        indices.extend(
            np.where(np.all(np.isclose(lattice_big, lattice[i], rtol=1e-3), axis=1))[0]
        )
    for i in indices:
        genome[i] = 1
    return genome, (n1, n2, n3, num_atoms)


def particles_encode_gen_lattice(lc, n1, n2, n3, num_atoms=None, max_num_atoms=None):
    num_lattice = max_num_atoms // 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    num_points = 0
    lattice_big, mean_position = empty_lattice(lc, n_max, n_max, n_max)
    indices = []
    lattice = empty_lattice(lc, n1, n2, n3, mean_position)
    num_points = len(lattice)
    genome = np.zeros(len(lattice_big))
    for i in range(num_atoms):
        indices.extend(
            np.where(np.all(np.isclose(lattice[i], lattice_big, rtol=1e-1), axis=1))[0]
        )
    for i in indices:
        genome[i] = 1
    return genome, (n1, n2, n3, num_atoms)


def recon_coordfromcodes(genome, lc, max_num_atoms):
    num_lattice = max_num_atoms // 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    lattice_big = empty_lattice(lc, n_max, n_max, n_max)
    index = list(np.where(np.array(genome) == 1)[0])
    # print(index)
    return np.array(lattice_big[index]), lattice_big


def encode_fromcoord(particle, lc, max_num_atoms):
    index = []
    num_lattice = max_num_atoms // 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    lattice = empty_lattice(lc, n_max, n_max, n_max)
    for i in range(len(particle)):
        index.extend(
            np.where(np.all(np.isclose(particle[i], lattice, rtol=1e-3), axis=1))[0]
        )
    # print(len(particle),len(index))
    genome_recon = np.zeros(len(lattice))
    for i in index:
        genome_recon[i] = 1
    return genome_recon


def ini_population(population_size, max_num_atoms, lc, num_atoms=None):
    generations = []
    parameters = []
    n = 0
    while n < population_size:
        genome, parameter = particles_encode_gen(
            lc, num_atoms=num_atoms, max_num_atoms=max_num_atoms
        )
        # no duplicates
        if parameter not in parameters:
            generations.append(genome)
            parameters.append(parameter)
            n += 1
    # oblate set
    assert len(set(parameters)) == len(parameters)
    return generations, parameters


def fitness(genome, lc, max_num_atoms, descriptors: dict):
    genome = np.array(genome)
    particle, _ = recon_coordfromcodes(genome, lc, max_num_atoms)
    atom = Atoms(positions=particle, symbols=["Pt"] * len(particle))
    cluster_descriptors = descriptor_table(atom)
    true_dis = np.array([descriptors[k] for k in descriptors.keys()])
    # print(true_dis)
    pred_dis = np.array([cluster_descriptors[k] for k in descriptors.keys()])
    # print(pred_dis)
    # MSE
    return np.sum((true_dis - pred_dis) ** 2), pred_dis


def parents(population, parameters, fitness_values, tol=1e5):
    fitness_values = np.array(fitness_values)
    ranking = ranking_fitness(fitness_values)
    prob = normalized_p(ranking)
    # index=np.arange(len(population))
    # threshold=random.uniform(np.min(prob),np.max(prob))
    current = 0
    index = 0
    # for i in range(len(population)):
    while index < tol:
        i = random.randint(0, len(population) - 1)
        index += 1
        if random.uniform(0, 1) <= prob[i]:
            return list(population[i]), parameters[i]
        if index == tol:
            raise Exception("exceed the maximum try times!")


def ranking_fitness(fitness_values):
    fitness_ranking = np.zeros(len(fitness_values))
    fitness_dict = Counter(fitness_values)
    fitness_ordered = sorted(fitness_dict.keys(), reverse=False)
    rank = 1
    for i in range(len(fitness_ordered)):
        indices = np.where(np.array(fitness_values) == fitness_ordered[i])[0]
        for j in indices:
            fitness_ranking[j] = rank
        rank += 1
    return fitness_ranking


def probability_selection(fitness_ranking):
    R = len(fitness_ranking)
    lamb = np.log(10) / (max(fitness_ranking) - min(fitness_ranking))
    # print(lamb)
    z = np.array([np.exp(-lamb * r) for r in fitness_ranking])
    z_sum = np.sum(z)
    return z / z_sum


def normalized_p(fitness_ranking):
    p = probability_selection(fitness_ranking)
    return (p - np.min(p)) / (np.max(p) - np.min(p))


def crossover(lc, parent1, parent2, cross_over_rate=0.6, max_num_atoms=100):
    if random.random() < cross_over_rate:
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        # parent1_particle=recon_coordfromcodes(parent1,lc,max_num_atoms)
        # parent2_particle=recon_coordfromcodes(parent2,lc,max_num_atoms)
        parent1_up, parent1_down = atom_cut_up_down_center(
            parent1, theta, phi, lc, max_num_atoms
        )
        parent2_up, parent2_down = atom_cut_up_down_center(
            parent2, theta, phi, lc, max_num_atoms
        )
        parent1 = parent1_up + parent2_down
        parent2 = parent2_up + parent1_down
        # print(parent1_up,parent2_down)
        # print(parent1_down,parent2_up)
        parent1[parent1 > 1] = 1
        parent2[parent2 > 1] = 1
        return parent1, parent2
    else:
        return parent1, parent2


# def crossover(lc,parent1,parent2,parameter1,parameter2,cross_over_rate=0.6,max_num_atoms=100):
#     num_atom1=parameter1[3]
#     num_atom2=parameter2[3]
#     min_num=min(num_atom1,num_atom2)


#     if random.random()<cross_over_rate:
#         crossover_point1=random.randint(1,min_num)
#         # crossover_point2=random.randint((len(parent1)-1)//2,len(parent1)-1)
#         # print(parent1)
#         # print(parent2)
#         # swith=random.choice([0,1])
#         return parent1[:crossover_point1]+parent2[crossover_point1:],parent2[:crossover_point1]+parent1[crossover_point1:],parameter1,parameter2
#         # if swith==0:
#         #     return parent1[:crossover_point1]+parent2[crossover_point1:crossover_point2]+parent1[crossover_point2:],parent2[:crossover_point1]+parent1[crossover_point1:crossover_point2]+parent2[crossover_point2:]
#         # if swith==1:
#         #     return parent1[crossover_point1:crossover_point2]+parent2[:crossover_point1]+parent1[crossover_point2:],parent2[crossover_point1:crossover_point2]+parent1[:crossover_point1]+parent2[crossover_point2:]
#     else:
#         return parent1,parent2,parameter1,parameter2
def mutation(genome, lc, max_num_atoms, mutation_rate=0.3):
    mr = random.random()
    num_atoms = len(np.where(np.array(genome) == 1)[0])
    if mr <= mutation_rate / 2:
        genome_update = np.zeros(len(genome))
        renum = 0
        while (
            len(np.where(np.array(genome_update) == 1)[0]) < 1 / 3 * num_atoms
            and renum < 5
        ):
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2 * np.pi)
            # if renum==0:
            #     print("no good 1")
            genome_update = atom_trancate_center(genome, theta, phi, lc, max_num_atoms)
            renum += 1
        if renum == 5:
            genome_update = genome
    if mr > mutation_rate / 2 and mr <= mutation_rate:
        genome_update = np.zeros(len(genome))
        renum = 0
        while (
            len(np.where(np.array(genome_update) == 1)[0]) < 1 / 3 * num_atoms
            and renum < 5
        ):
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2 * np.pi)
            genome_update = atom_trancate_arbi(genome, theta, phi, lc, max_num_atoms)
            renum += 1
        if renum == 5:
            genome_update = genome

    else:
        genome_update = genome
    return genome_update


def atom_cut_up_down_center(genome, theta, phi, lc, max_num_atoms):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

    num_atoms = len(np.where(np.array(genome) == 1)[0])
    particle, big_lattice = recon_coordfromcodes(genome, lc, max_num_atoms)
    particle = np.array(particle)
    center = np.mean(particle, axis=0)
    closest_index = np.where(
        np.all(np.isclose(center, particle, rtol=lc / (1.8 * np.sqrt(2))), axis=1)
    )[0][0]
    particle_shift = particle - particle[closest_index]
    center = np.mean(particle_shift, axis=0)
    normal = normal_vector(center, theta, phi)

    trancate_up_particle = []
    trancate_down_particle = []
    for pos in particle_shift:
        if normal[0] * (pos[0] - center[0]) + normal[1] * (pos[1] - center[1]) + normal[
            2
        ] * (pos[2] - center[2]) >= -lc / np.sqrt(2):
            trancate_up_particle.append([pos[0], pos[1], pos[2]])
        if normal[0] * (pos[0] - center[0]) + normal[1] * (pos[1] - center[1]) + normal[
            2
        ] * (pos[2] - center[2]) <= lc / np.sqrt(2):
            trancate_down_particle.append([pos[0], pos[1], pos[2]])
    # print(len(trancate_up_particle))
    # print(len(trancate_down_particle))
    genome_up_recon = encode_fromcoord(trancate_up_particle, lc, max_num_atoms)
    genome_down_recon = encode_fromcoord(trancate_down_particle, lc, max_num_atoms)
    return np.array(genome_up_recon), np.array(genome_down_recon)


def atom_trancate_center(genome, theta, phi, lc, max_num_atoms):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

    num_atoms = len(np.where(np.array(genome) == 1)[0])
    particle, big_lattice = recon_coordfromcodes(genome, lc, max_num_atoms)
    particle = np.array(particle)
    center = np.mean(particle, axis=0)
    normal = normal_vector(center, theta, phi)
    trancate_particle = []
    for pos in particle:
        if (
            normal[0] * (pos[0] - center[0])
            + normal[1] * (pos[1] - center[1])
            + normal[2] * (pos[2] - center[2])
            >= 0
        ):
            trancate_particle.append([pos[0], pos[1], pos[2]])
    if len(trancate_particle) < 5:
        genome_recon = np.zeros(len(genome))
    else:
        genome_recon = encode_fromcoord(trancate_particle, lc, max_num_atoms)
    # print(f"1={len(genome_recon)}")
    # print(len(particle),len(trancate_particle),len(np.where(np.array(genome_recon)==1)[0]))
    return genome_recon


def atom_trancate_arbi(genome, theta, phi, lc, max_num_atoms):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

    num_atoms = len(np.where(np.array(genome) == 1)[0])
    particle, big_lattice = recon_coordfromcodes(genome, lc, max_num_atoms)
    particle = np.array(particle)
    center_get = []
    for i in range(len(particle)):
        CN1, dis = getCN_dis_Oneshell(particle, particle[i], 1, thickness=0.1)
        if CN1 == 12:
            center_get.append(particle[i])
    if len(center_get) != 0:
        center = random.choice(center_get)
    else:
        return particle
    ## points with coordination numbers are 12 are considered as the center of planes for truncation.
    normal = normal_vector(center, theta, phi)
    trancate_particle = []
    for pos in particle:
        if (
            normal[0] * (pos[0] - center[0])
            + normal[1] * (pos[1] - center[1])
            + normal[2] * (pos[2] - center[2])
            >= 0
        ):
            trancate_particle.append([pos[0], pos[1], pos[2]])
    # print(f"len(trancate_particle)={len(trancate_particle)}")
    if len(trancate_particle) < 5:
        trancate_particle = []
        for pos in particle:
            if (
                normal[0] * (pos[0] - center[0])
                + normal[1] * (pos[1] - center[1])
                + normal[2] * (pos[2] - center[2])
                <= 0
            ):
                trancate_particle.append([pos[0], pos[1], pos[2]])
        # print(f"len(trancate_particle)={len(trancate_particle)}")

    genome_recon = encode_fromcoord(trancate_particle, lc, max_num_atoms)
    return genome_recon


def convex_particle(genome, shell, lc, max_num_atoms):
    encodes = genome.copy()
    atom_num = len(np.where(encodes == 1)[0])
    CN_steps = []
    index_particle = []
    zero_dict = dict()
    control = 0
    while control < 100:
        CNs1 = []
        dis1 = []
        index_particle = []
        particle, big_lattice = recon_coordfromcodes(encodes, lc, max_num_atoms)
        for i in range(len(big_lattice)):
            if encodes[i] == 1:
                CN1, dis = getCN_dis_Oneshell(
                    particle, big_lattice[i], shell, thickness=0.1
                )
                if dis <= np.round(np.sqrt(2) / 2 * lc, 4) and CN1 > 4:
                    index_particle.append(i)
                else:
                    encodes[i] = 0

        for i in range(len(big_lattice)):
            if encodes[i] == 0:
                CN0, dis = getCN_dis_Oneshell(
                    particle, big_lattice[i], shell, thickness=0.1
                )
                if CN0 != 0 and dis <= np.round(np.sqrt(2) / 2 * lc, 4):
                    zero_dict[i] = CN0
        zero_dict = dict(
            sorted(zero_dict.items(), key=lambda item: item[1], reverse=True)
        )
        residual = atom_num - len(index_particle)
        for i in range(residual):
            encodes[list(zero_dict.keys())[i]] = 1
        # print(f"particle {j}")
        particle, big_lattice = recon_coordfromcodes(encodes, lc, max_num_atoms)
        atom = Atoms(positions=particle, symbols=["Pt"] * len(particle))
        _, _, CN_ave = getCN_dis_N(atom, shell)
        control += 1
        CN_steps.append(CN_ave)
        if control > 10:
            if CN_steps[-1] - CN_steps[-2] < 1e-3:
                break
    return encodes


def genetic_algorithm(
    max_num_atoms=200,
    generations=50,
    population_size=500,
    lc=3.77,
    mutation_rate=0.3,
    cross_over_rate=0.6,
    shell=1,
    descriptors={
        "surface ratio": 1,
        #  "flatten": 0.7,
        "atom_number": 100
        # "CN1":6,
        # "CN2":1.5,
        # "CN3":2,
        # "CN4":5
    },
):
    # 1. set up parameters

    print(descriptors)
    num_lattice = max_num_atoms // 2  # since one unit cell of fcc lattice has 2 atoms
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    lattice_big = empty_lattice(lc, n_max, n_max, n_max)

    # 2. initialization of populations
    populations, parameters = ini_population(
        lc=lc,
        population_size=population_size,
        max_num_atoms=max_num_atoms,
        num_atoms=None,
    )
    # populations_ini_mute=[]
    # for i in range(len(populations)):
    #     populations_ini_mute.append(mutation(populations[i],lc,max_num_atoms,mutation_rate=0.5))
    ini_population_record = populations.copy()
    fitness_record = []
    best_particle = []
    # 3. calculate fitness for populations
    fitness_values = [
        fitness(genome, lc, max_num_atoms, descriptors)[0] for genome in populations
    ]
    # 4. start iteration
    for generation in tqdm(range(generations)):
        # clustering_energy_values=[clustering_energy(genome,lattice) for genome in population]
        # (1). generate new populations
        new_populations = []

        # check duplicates, if duplicates exist, then generate a new structure
        for _ in range(population_size // 2):
            # [1]. find parents from populations based on fitness values
            # print(len(populations),len(parameters),len(fitness_values))
            parent1, parameters1 = parents(populations, parameters, fitness_values)
            parent2, parameters2 = parents(populations, parameters, fitness_values)
            fit_parent1 = fitness(parent1, lc, max_num_atoms, descriptors)
            fit_parent2 = fitness(parent2, lc, max_num_atoms, descriptors)
            # [2]. crossover and mutation to generate offsprings
            children1, children2 = crossover(
                lc,
                parent1,
                parent2,
                cross_over_rate=cross_over_rate,
                max_num_atoms=max_num_atoms,
            )
            # print(f"children={len(np.where(np.array(children1)==1)[0]),len(np.where(np.array(children2)==1)[0])}")
            children1 = mutation(
                children1, lc, max_num_atoms, mutation_rate=mutation_rate
            )
            children2 = mutation(
                children2, lc, max_num_atoms, mutation_rate=mutation_rate
            )
            # children1=convex_particle(children1,shell,lc,max_num_atoms)
            # children2=convex_particle(children2,shell,lc,max_num_atoms)
            new_populations.extend([children1, children2])

        # (2). calculate fitness for new populations and obtain the predicted descriptors
        fitness_values = []
        predict_values = []
        for genome in new_populations:
            # print(f"genome={genome}")
            fitness_cal = fitness(genome, lc, max_num_atoms, descriptors)
            fitness_values.append(fitness_cal[0])
            predict_values.append(fitness_cal[1])
        # clustering_energy_values=[clustering_energy(genome,lattice) for genome in population]
        # (3). find the best solution
        # [1]. best fitness value and prediction
        best_fitness = min(fitness_values)
        ave_fitness = np.mean(fitness_values)
        best_predict = predict_values[fitness_values.index(best_fitness)]
        best_index = fitness_values.index(best_fitness)
        # [2]. reconstruct the best solution from codes using new_populations.
        particle, _ = recon_coordfromcodes(
            new_populations[best_index], lc, max_num_atoms
        )
        # best_clustering_energy=min(clustering_energy_values)
        # [3]. record the reconstructed particles using Atoms objects.
        best_particle.append(Atoms(positions=particle, symbols=["Pt"] * len(particle)))
        fitness_record.append(ave_fitness)  # dong
        # [4]. print output
        print(
            f"Generation{generation}:ave Finess={ave_fitness},best Predict={best_predict}"
        )
        # [5]. update populations
        populations = new_populations
        # [6]. early stopping
        # if generation>10:
        #     error=fitness_record[generation-1]-fitness_record[generation]
        #     if error<1e-4:
        #         break
    # 4. record the best solution
    best_index = fitness_values.index(best_fitness)
    print(f"best_index={best_index}")
    best_solution = populations[best_index]
    # print(len(population))
    index_best = np.where(np.array(best_solution) == 1)[0]
    best_atom = Atoms(
        positions=lattice_big[index_best], symbols=["Pt"] * len(lattice_big[index_best])
    )
    last_population = populations
    last_atom_list = []
    for i in range(len(last_population)):
        particle, _ = recon_coordfromcodes(last_population[i], lc, max_num_atoms)
        last_atom_list.append(Atoms(positions=particle, symbols=["Pt"] * len(particle)))
    ini_atom_list = []
    for i in range(len(ini_population_record)):
        particle, _ = recon_coordfromcodes(ini_population_record[i], lc, max_num_atoms)
        ini_atom_list.append(Atoms(positions=particle, symbols=["Pt"] * len(particle)))
    return (
        best_atom,
        fitness_record,
        fitness_values,
        best_particle,
        last_atom_list,
        ini_atom_list,
        last_population,
    )


# def genome(num_atoms):
#     np.random
if __name__ == "__main__":
    max_num_atoms = (200,)
    generations = (50,)
    population_size = (500,)
    lc = (3.77,)
    mutation_rate = (0.3,)
    cross_over_rate = (0.6,)
    shell = (1,)
    descriptors = {
        "surface ratio": 1,
        #  "flatten": 0.7,
        "atom_number": 100
        # "CN1":6,
        # "CN2":1.5,
        # "CN3":2,
        # "CN4":5
    }
    (
        best_atom,
        fitness_record,
        fitness_values,
        best_particle,
        last_atom_list,
        ini_atom_list,
        population,
    ) = genetic_algorithm(
        max_num_atoms=max_num_atoms,
        generations=generations,
        population_size=population_size,
        lc=lc,
        mutation_rate=mutation_rate,
        cross_over_rate=cross_over_rate,
        shell=shell,
        descriptors=descriptors,
    )
