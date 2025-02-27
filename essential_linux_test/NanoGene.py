from itertools import *
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.visualize import view
from random import sample, seed
from shape_proj_util.geo_tools.geometry import *
# from ase.io.xyz import write_xyz
from ase.io import write
from scipy.spatial.distance import cdist
import random
from tqdm.auto import tqdm
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree
# from ase.io.x3d import write_x3d
import os
import toml
from time import time
#import warnings
import plotly.graph_objects as go
#warnings.filterwarnings("error")

def fccbasis(a):
    fcc = [
        [0, 0.5 * a, 0.5 * a],
        [0.5 * a, 0, 0.5 * a],
        [0.5 * a, 0.5 * a, 0],
        [0, 0, 0],
    ]
    return np.array(fcc)


# lattice is an array containing coordinates in fcc
def extendfcc(lattice, a, x, y, z):
    lattice_extend = []
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                for o in range(0, len(lattice)):
                    if not lattice_extend.__contains__(
                        [
                            lattice[o][0] + i * a,
                            lattice[o][1] + j * a,
                            lattice[o][2] + k * a,
                        ]
                    ):
                        lattice_extend.extend(
                            [
                                [
                                    lattice[o][0] + i * a,
                                    lattice[o][1] + j * a,
                                    lattice[o][2] + k * a,
                                ]
                            ]
                        )
    return np.array(lattice_extend)


def empty_lattice(lc, n1, n2, n3, lattice_big):
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
    lattice = np.round(lattice - mid_position, 6)
    x_max, y_max, z_max = np.max(lattice, axis=0)
    x_min, y_min, z_min = np.min(lattice, axis=0)
    # lattice=np.round(lattice-mid_position,6)
    # lattice=order_pos(lattice)
    lattice = lattice[
        np.where(
            (lattice[:, 0] >= x_min)
            & (lattice[:, 0] <= x_max)
            & (lattice[:, 1] >= y_min)
            & (lattice[:, 1] <= y_max)
            & (lattice[:, 2] >= z_min)
            & (lattice[:, 2] <= z_max)
        )
    ]
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


def particles_encode_gen_no_oblate(lc, lattice_big, num_atoms=None, max_num_atoms=None):
    if num_atoms == None:
        assert (
            max_num_atoms != None
        ), "Please provide the maximum number of atoms in the lattice."
        num_atoms = random.randint(12, max_num_atoms)
    # min_num_points=num_atoms
    num_lattice = max_num_atoms #//2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    num_points = 0
    # transfer points to the big lattice
    indices = []
    # genome=np.zeros(len(lattice_big))
    for i in range(num_atoms):
        indices.extend(
            np.where(
                np.all(np.isclose(lattice_big, lattice_big[i], rtol=1e-3), axis=1)
            )[0]
        )
    return lattice_big[indices]


# generate random lattice with random n1,n2,n3 of the superlattice parimeters, and random numbers of atoms in the lattice.
def particles_encode_gen(lc, lattice_big, num_atoms=None, max_num_atoms=None):
    if num_atoms == None:
        # assert max_num_atoms!=None,"Please provide the maximum number of atoms in the lattice."
        num_atoms = random.randint(12, max_num_atoms)
    num_lattice = max_num_atoms #// 2
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    # min_num_points=num_atoms

    # print(f"length={np.mean(lattice_big,axis=0)}")
    # times=random.randint(1,3)
    # the extreme case is that the particle has only one layer of unit cells, which has the dimension: np.sqrt(n_max**3)*np.sqrt(n_max**3)*1
    n1 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
    n2 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
    n3 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
    lattice = empty_lattice(lc, n1, n2, n3, lattice_big)
    num_points = len(lattice)
    # this is too strong
    while num_points < num_atoms * 2 or n1 * n2 * n3 > n_max**3:
        n1 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
        n2 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
        n3 = random.randint(1, np.int32(np.floor(np.sqrt(n_max**3))))
        lattice = empty_lattice(lc, n1, n2, n3, lattice_big)
        num_points = len(lattice)
    lattice = lattice[:num_atoms]
    # print((num_points,n1,n2,n3,num_atoms))
    # print(f"length={np.mean(lattice,axis=0)}")
    # transfer points to the big lattice

    return lattice


def ini_population(
    lc, population_size, lattice_big, num_atoms=None, max_num_atoms=None
):
    generations = []
    n = 0
    while n < population_size:
        particle = particles_encode_gen(
            lc,
            lattice_big=lattice_big,
            num_atoms=num_atoms,
            max_num_atoms=max_num_atoms,
        )
        # print(particle)
        generations.append(particle)
        n += 1
    return generations


def fitness(particle, descriptors: dict, fitness_func="L1", reduction="sum",weight:np.ndarray=None):
    if weight==None:
        weight=np.array([1]*len(descriptors.keys()))
    atom = Atoms(positions=particle, symbols=["Pt"] * len(particle))
    try:
        if "oblateness_moment" not in descriptors.keys():
            cluster_descriptors = descriptor_table(
            atom, all=False, descriptors=list(descriptors.keys()) + ["oblateness_moment"]
            )
        else:
            cluster_descriptors = descriptor_table(
                atom,all=False,descriptors=list(descriptors.keys())
            )
    except RuntimeWarning:
        print(f"calculate descriptor: {atom.get_positions()}")
    true_dis = []
    for k in descriptors.keys():
        if k != "ellipsoid":
            true_dis.append(descriptors[k])
        else:
            true_dis.append(descriptors[k][0])
            true_dis.append(descriptors[k][1])
            true_dis.append(descriptors[k][2])
    true_dis = np.array(true_dis)
    # print(true_dis)
    if "oblateness_moment" not in descriptors.keys():
        pred_dis = np.array(
            [cluster_descriptors[k] for k in list(cluster_descriptors.keys())[:-1]]
        )
    else:
        pred_dis=np.array([cluster_descriptors[k] for k in list(cluster_descriptors.keys())])
    # print(pred_dis)
    # MSE
    # np.sum((true_dis-pred_dis)**2)
    sim = 0
    if cluster_descriptors["oblateness_moment"] > 1:
        if fitness_func=="L1":
            sim = np.abs(true_dis - pred_dis)
        elif fitness_func=="L2":
            sim = np.power(true_dis - pred_dis, 2)
        elif fitness_func=="RAE":
            sim = (np.abs(true_dis - pred_dis) / true_dis)

        if reduction=="sum":
            sim = np.dot(sim,weight) * 10 ** cluster_descriptors["oblateness_moment"]
        elif reduction=="mean":
            sim = np.dot(sim,weight)/np.sum(weight) * 10 ** cluster_descriptors["oblateness_moment"]
        elif reduction=="max":
            sim = np.max(np.multiply(sim,weight)) * 10 ** cluster_descriptors["oblateness_moment"]
    else:
        if fitness_func=="L1":
            sim = np.abs(true_dis - pred_dis)
            # print(sim)
        elif fitness_func=="L2":
            sim = np.power(true_dis - pred_dis, 2)
        elif fitness_func=="RAE":
            sim = (np.abs(true_dis - pred_dis) / true_dis)
        if reduction=="sum":
            sim = np.sum(sim) 
        elif reduction=="mean":
            sim = np.mean(sim) 
        elif reduction=="max":
            sim = np.max(sim)
    return sim, pred_dis


def parents(population, fitness_values, tol=1e5):
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
            return list(population[i])
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


def atom_cut_up_down_center(particle, theta, phi, lc):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

    num_atoms = len(particle)
    particle = np.array(particle)
    if len(particle) < 5:
        print("Warning particle is less than 5 atoms")
        return particle, particle
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
    return np.array(trancate_up_particle), np.array(trancate_down_particle)


def atom_trancate_center(particle, theta, phi):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

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
    if len(trancate_particle)<5:
        trancate_particle=[]
        for pos in particle:
                if normal[0]*(pos[0]-center[0])+normal[1]*(pos[1]-center[1])+normal[2]*(pos[2]-center[2])<0:
                      trancate_particle.append([pos[0],pos[1],pos[2]])


    # print(f"1={len(genome_recon)}")
    # print(len(particle),len(trancate_particle),len(np.where(np.array(genome_recon)==1)[0]))
    return trancate_particle


def atom_trancate_arbi(particle, theta, phi):
    def normal_vector(center, theta, phi):
        return np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

    num_atoms = len(particle)
    particle = np.array(particle)
    center_get = np.mean(particle, axis=0)
    particle_vote = particle.copy()
    if [center_get] not in particle_vote:
        np.vstack((particle_vote, center_get))
    # for i in range(len(particle)):
    #     CN1,dis=getCN_dis_Oneshell(particle,particle[i],1,thickness=0.1)
    #     if CN1==12:
    #         center_get.append(particle[i])
    # if len(center_get)!=0:
    center = random.choice(particle_vote)
    # else:
    #     return particle
    ## points with coordination numbers are 12 are considered as the center of planes for truncation.
    normal = normal_vector(center, theta, phi)
    trancate_particle = []
    for pos in particle:
        if (
            normal[0] * (pos[0] - center[0])
            + normal[1] * (pos[1] - center[1])
            + normal[2] * (pos[2] - center[2])
            >= 0 and [pos[0], pos[1],pos[2]] not in trancate_particle
        ):
            trancate_particle.append([pos[0], pos[1], pos[2]])
    # print(f"len(trancate_particle)={len(trancate_particle)}")
        if len(trancate_particle)<5:
             trancate_particle=[]
             for pos in particle:
                if normal[0]*(pos[0]-center[0])+normal[1]*(pos[1]-center[1])+normal[2]*(pos[2]-center[2])<=0:
                      trancate_particle.append([pos[0],pos[1],pos[2]])
    # print(f"len(trancate_particle)={len(trancate_particle)}")
    return trancate_particle


def crossover(lc, parent1, parent2, cross_over_rate=0.6):
    def remove_duplicates(particle, lc):
        duplicates = []
        index_get = []
        for i in range(len(particle)):
            index = np.where(
                np.isclose(cdist([particle[i]], particle)[0], 0, atol=1e-2)
            )[0]
            if len(index) > 1:
                duplicates.extend(index[1:])
            if index[0] not in duplicates:
                index_get.append(index[0])
        return np.array([particle[i] for i in index_get])

    if random.random() < cross_over_rate:
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        # parent1_particle=recon_coordfromcodes(parent1,lc,max_num_atoms)
        # parent2_particle=recon_coordfromcodes(parent2,lc,max_num_atoms)
        try:
            parent1_up, parent1_down = atom_cut_up_down_center(parent1, theta, phi, lc)
            parent2_up, parent2_down = atom_cut_up_down_center(parent2, theta, phi, lc)
        except:
            return parent1, parent2
        parent1 = np.vstack((parent1_up, parent2_down))
        parent2 = np.vstack((parent1_down, parent2_up))
        parent1 = remove_duplicates(parent1, lc)
        parent2 = remove_duplicates(parent2, lc)
        # print(parent1_up,parent2_down)
        # print(parent1_down,parent2_up)
        return parent1, parent2
    else:
        return parent1, parent2


def mutation(particle, lc, lattice_big, max_num_atoms, mutation_rate=0.3):
    mr = random.uniform(0, 1)
    num_atoms = len(particle)

    # if mr <= mutation_rate/3:  # / 2:
    #     # renum = 0
    #     # theta = random.uniform(0, np.pi)
    #     # phi = random.uniform(0, 2 * np.pi)
    #     # particle_update = atom_trancate_center(particle, theta, phi)
    #     # while len(particle_update) < 5 and renum < 5:
    #     #     theta = random.uniform(0, np.pi)
    #     #     phi = random.uniform(0, 2 * np.pi)
    #     #     # if renum==0:
    #     #     #     print("no good 1")
    #     #     particle_update = atom_trancate_center(particle, theta, phi)
    #     #     renum += 1
    if mr <= mutation_rate / 2:  # and mr <= mutation_rate * 2 / 3:
        renum = 0
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        particle_update = atom_trancate_arbi(particle, theta, phi)
        while len(particle_update) < 5 and renum < 5:
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2 * np.pi)
            particle_update = atom_trancate_arbi(particle, theta, phi)
            renum += 1
    elif mr > mutation_rate * 1 / 2 and mr <= mutation_rate:
        # 2. initialization of populations
        particle_update = ini_population(
            lc=lc,
            population_size=1,
            lattice_big=lattice_big,
            max_num_atoms=max_num_atoms,
            num_atoms=None,
        )[0]

    else:
        particle_update = particle
    return particle_update


def convex_particle(particle, lattice_big, lc):
    indices = []
    try:
        hull = ConvexHull(particle)
    except QhullError:
        # print("less than 3 dimensions")
        # print(particle)
        return particle
    for points in lattice_big:
        if is_point_in_hull(points, hull, lc):
            indices.extend(
                np.where(np.all(np.isclose(points, lattice_big, rtol=1e-1), axis=1))[0]
            )
    # print(len(indices))
    # print(len(particle))
    return np.array([lattice_big[i] for i in indices])


def is_point_in_hull(point, hull, lc):
    equations = hull.equations
    # print(np.all(np.dot(equations[:,:-1],point)+equations[:,-1]<=0))
    return np.all(
        np.dot(equations[:, :-1], point) + equations[:, -1] <= lc / (2 * np.sqrt(2))
    )


def find_most_similar(population, individual, crowding_distance=0.1):
    return np.array(
        [
            min(
                population,
                key=lambda x: calculate_similarity(x, individual, crowding_distance),
            )
        ]
    )[0]


def chamfer_distance(point_cloud_A, point_cloud_B):
    tree_A = cKDTree(point_cloud_A)
    tree_B = cKDTree(point_cloud_B)

    dist_A_to_B, _ = tree_A.query(point_cloud_B, k=1)
    dist_B_to_A, _ = tree_B.query(point_cloud_A, k=1)

    chamfer_dist = np.mean(dist_A_to_B) + np.mean(dist_B_to_A)

    return chamfer_dist


def is3Ddimension(particle):
    a=len(np.unique(particle[:,0]))
    b=len(np.unique(particle[:,1]))
    c=len(np.unique(particle[:,2]))
    if a==1 or b==1 or c==1:
        return False
    else:
        return True



def align_point_clouds_pca(particle1, particle2):
    # align particle1 to particle2
    # Center the point clouds
    source_centered = particle1 - np.mean(particle1, axis=0)
    target_centered = particle2 - np.mean(particle2, axis=0)




    # Perform PCA
    if is3Ddimension(source_centered) and is3Ddimension(target_centered):
        try:
            source_pca = PCA(n_components=3).fit(source_centered)
        except RuntimeWarning:
            print(f"align_pca(source):{source_centered}")
        try:
            target_pca = PCA(n_components=3).fit(target_centered)
        except RuntimeWarning:
            print(f"align_pca(target):{target_centered}")
    else:
        return particle1


    # Align principal components
    R = np.dot(target_pca.components_.T, source_pca.components_)
    aligned_source = np.dot(source_centered, R.T) + np.mean(particle2, axis=0)

    return aligned_source


def sharing_function(ind1, ind2, niche_radius, alpha):
    distance = calculate_similarity(ind1, ind2, niche_radius)
    if distance < niche_radius:
        return 1 - (distance / niche_radius) ** alpha
    else:
        return 0


def calculate_similarity(ind1, ind2, crowding_distance=0.1):
    # atoms1 = Atoms(positions=ind1, symbols=["Pt"] * len(ind1))
    # atoms2 = Atoms(positions=in2, symbols=["Pt"] * len(in2))
    # des_atoms1 = descriptor_table(atoms1)
    # score1 = np.array(np.array([des_atoms1[k] for k in des_atoms1.keys()]))
    # des_atoms2 = descriptor_table(atoms2)
    # score2 = np.array([des_atoms2[k] for k in des_atoms2.keys()])
    # return np.mean((score1 - score2) ** 2)
    ind1 = align_point_clouds_pca(ind1, ind2)
    dist = chamfer_distance(ind1, ind2)

    return dist / crowding_distance


def shared_fitness_value(individual, descriptor, population, niche_radius, alpha,fitness_func,reduction,weight):
    fitnessvalue, predict_value = fitness(individual, descriptor,fitness_func,reduction,weight)
    sharing_sum = sum(
        sharing_function(individual, other, niche_radius, alpha) for other in population
    )
    return fitnessvalue / sharing_sum, predict_value


def fitness_sharing(population, descriptor, niche_radius, alpha,fitness_func="L1",reduction="sum",weight=None):
    shared_fitness = []
    predict_values = []
    for individual in population:
        fitness_value, predict_value = shared_fitness_value(
            individual, descriptor, population, niche_radius, alpha,fitness_func,reduction,weight
        )
        shared_fitness.append(fitness_value)
        predict_values.append(predict_value)
    return shared_fitness, predict_values


def write_file(Atoms_list, foldername):
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    n = 1
    for atoms in Atoms_list:
        write(f"{foldername}/individual_{n}.xyz", atoms, format="xyz")
        n += 1


def draw_ellipsoid(atoms, save=False, filename="ellipsoid.png"):
    position = atoms.get_positions()
    radii = ellipsoid(atoms)
    center = radii[3]
    rotation = radii[4]
    # Create a grid of points
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Create the ellipsoid points in the local space
    x = radii[0] * np.outer(np.sin(v), np.cos(u))
    y = radii[1] * np.outer(np.sin(v), np.sin(u))
    z = radii[2] * np.outer(np.cos(v), np.ones_like(u))

    for i in range(len(x)):
        for j in range(len(x[0])):
            # print(np.dot([x[i, j], y[i, j], z[i, j]], radii) * rotation + center)
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot(rotation, [x[i, j], y[i, j], z[i, j]]) + center
            )

    # Plot ellipsoid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color="cyan", alpha=0.3)
    ax.scatter(position[:, 0], position[:, 1], position[:, 2], color="red")
    fig.savefig(filename)


def genetic_algorithm(
    max_num_atoms=200,
    generations=40,
    population_size=100,
    lc=3.77,
    cross_over_rate=0.3,
    elite_fraction=0.05,
    initial_crowding_distance=0.1,
    percentage=0.75,
    niche_radius=0.1,
    alpha=1,
    initial_mutation_rate=1,
    mutation_rate_decay=0.9,
    weight=None,
    fitness_func="L1",
    reduction="sum",
    descriptors={"oblateness_moment": 1, "atom_number": 55}
    # "CN1":6,
    # "CN2":1.5,
    # "CN3":2,
    # "CN4":5
):
    # 1. set up parameters

    print(descriptors)
    num_lattice = max_num_atoms  # number of lattice on each direction equals cube root of the maximum number of atoms minus 1(\cbrt(max_num_atoms)-1)
    n_max = int(np.ceil(np.cbrt(num_lattice)))-1
    lattice_big = extendfcc(
        fccbasis(lc),
        lc,
        int(np.sqrt(n_max**3)),
        int(np.sqrt(n_max**3)),
        int(np.sqrt(n_max**3)),
    )
    lattice_big = lattice_big - np.mean(lattice_big, axis=0)

    # 2. initialization of populations
    populations = ini_population(
        lc=lc,
        population_size=population_size,
        lattice_big=lattice_big,
        num_atoms=None,
        max_num_atoms=max_num_atoms,
    )
    mutation_rate = initial_mutation_rate
    # populations_ini_mute=[]
    # for i in range(len(populations)):
    #     populations_ini_mute.append(mutation(populations[i],lc,max_num_atoms,mutation_rate=0.5))
    ini_population_record = populations.copy()
    fitness_record = []
    time_cost_record=[]
    # best_particle = []
    # 3. calculate fitness for populations
    fitness_values = fitness_sharing(populations, descriptors, niche_radius, alpha,fitness_func=fitness_func,reduction=reduction,weight=weight)[0]
    # print(len(fitness_values),len(populations))
    # 4. start iteration
    for generation in (pbar := tqdm(range(generations))):
        time_start=time()
        # clustering_energy_values=[clustering_energy(genome,lattice) for genome in population]
        # (1). generate new populations
        pbar.set_description(f"mutation_rate={np.round(mutation_rate,5)}")
        new_populations = []
        fit_pop = dict()
        for i in range(len(populations)):
            fit_pop[fitness_values[i]] = populations[i]
        sorted_population = [
            sorted(fit_pop.items(), key=lambda item: item[0], reverse=False)[i][1]
            for i in range(len(fit_pop))
        ]
        # print(population_size,elite_fraction)
        num_elites = int(population_size * elite_fraction)
        elites = sorted_population[:num_elites]
        # check duplicates, if duplicates exist, then generate a new structure
        num_residual = population_size - num_elites
        n = 0
        while n != num_residual:
            # [1]. find parents from populations based on fitness values
            # print(len(populations),len(parameters),len(fitness_values))
            parent1 = parents(populations, fitness_values)
            parent2 = parents(populations, fitness_values)
            parent1 = np.array(parent1)
            parent2 = np.array(parent2)
            # fit_parent1=fitness(parent1,lc,max_num_atoms,descriptors)
            # fit_parent2=fitness(parent2,lc,max_num_atoms,descriptors)
            # [2]. crossover and mutation to generate offsprings
            children1, children2 = crossover(
                lc, parent1, parent2, cross_over_rate=cross_over_rate
            )

            # print(f"children={len(np.where(np.array(children1)==1)[0]),len(np.where(np.array(children2)==1)[0])}")
            children1 = mutation(
                children1, lc, lattice_big, max_num_atoms, mutation_rate=mutation_rate
            )
            children2 = mutation(
                children2, lc, lattice_big, max_num_atoms, mutation_rate=mutation_rate
            )
            # convex hull

            children1 = convex_particle(children1, lattice_big, lc)
            children2 = convex_particle(children2, lattice_big, lc)
            if len(children1) == 0 or len(children2) == 0:
                continue
            # children1=convex_particle(children1,shell,lc,max_num_atoms)
            # children2=convex_particle(children2,shell,lc,max_num_atoms)
            new_populations.extend([children1, children2])
            n += 2
            if num_residual - n == 1:
                parent1 = parents(populations, fitness_values)
                parent2 = parents(populations, fitness_values)
                parent1 = np.array(parent1)
                parent2 = np.array(parent2)
                children1, children2 = crossover(
                    lc, parent1, parent2, cross_over_rate=cross_over_rate
                )
                children1 = mutation(
                    children1,
                    lc,
                    lattice_big,
                    max_num_atoms,
                    mutation_rate=mutation_rate,
                )
                children2 = mutation(
                    children2,
                    lc,
                    lattice_big,
                    max_num_atoms,
                    mutation_rate=mutation_rate,
                )
                children1 = convex_particle(children1, lattice_big, lc)
                children2 = convex_particle(children2, lattice_big, lc)
                if len(children1) == 0 or len(children2) == 0:
                    continue
                new_populations.extend([children1])
                n += 1
        new_populations.extend(elites)
        crowding_distance = initial_crowding_distance
        for child in new_populations:
            try:
                most_similar_individual = find_most_similar(
                    populations, child, crowding_distance
                )
                if (
                    fitness(child, descriptors)[0]
                    <= fitness(most_similar_individual, descriptors)[0]
                ):
                    index = 0
                    for ind in populations:
                        if np.array_equal(ind, most_similar_individual):
                            break
                        index += 1
                    populations.pop(index)
                    populations.append(child)
            except:
                continue
        # for child in new_populations:
        #     print(f"child={populations}")
        #     most_similar_individual = find_most_similar(populations, child)
        #     # print(f"most_similar_individual={most_similar_individual.shape}")
        #     # print(f"child={child.shape}")

        #     if fitness(child,descriptors)[0] <= fitness(most_similar_individual,descriptors)[0]:

        #         populations = [ind for ind in populations if not np.array_equal(ind, most_similar_individual)]
        #         new_population2.append(child)
        #     else:
        #         new_population2.append(most_similar_individual)
        # new_population2 = new_population2[:population_size]
        # new_populations.extend(elites)
        # print(f"new_populations={len(new_populations)}")

        # (2). calculate fitness for new populations and obtain the predicted descriptors
        fitness_values = []
        predict_values = []
        # for particle in populations:
        # print(f"genome={genome}")
        # fitness_cal = fitness(particle, descriptors)
        fitness_values, predict_values = fitness_sharing(
            populations, descriptors, initial_crowding_distance, alpha,fitness_func,reduction
        )
        # fitness_values.append(fitness_cal[0])
        # predict_values.append(fitness_cal[1])
        # clustering_energy_values=[clustering_energy(genome,lattice) for genome in population]
        # (3). find the best solution
        # [1]. best fitness value and prediction
        best_fitness = min(fitness_values)
        ave_fitness = np.mean(fitness_values)
        # std_fitness = np.std(fitness_values)
        # best_predict = predict_values[fitness_values.index(best_fitness)]
        fitness_value_sort = sorted(fitness_values)
        quater_fitness_index = np.where(
            np.array(fitness_values)
            <= fitness_value_sort[int(len(fitness_values) * percentage)-1]
        )[0][0]
        predict_quater = predict_values[quater_fitness_index]

        best_index = fitness_values.index(best_fitness)
        mean_predict = np.mean(predict_values)
        # [2]. reconstruct the best solution from codes using new_populations.
        particle = populations[best_index]
        # best_clustering_energy=min(clustering_energy_values)
        # [3]. record the reconstructed particles using Atoms objects.
        # best_particle.append(Atoms(positions=particle, symbols=["Pt"] * len(particle)))
        fitness_record.append(ave_fitness)  # dong
        mutation_rate *= mutation_rate_decay
        time_end = time()
        time_cost = time_end - time_start
        time_cost_record.append(time_cost)
        # print(mutation_rate)
        # [4]. print output
        print(
            f"Generation{generation}: Finess={np.round(np.mean(fitness_values),3)}+/-{np.round(np.std(fitness_values),3)} Predict_{percentage}={predict_quater} time_cost={time_cost}"
        )

        # [6]. early stopping
        # if generation>10:
        #     error=fitness_record[generation-1]-fitness_record[generation]
        #     if error<1e-4:
        #         break
    # 4. record the best solution
    best_index = fitness_values.index(best_fitness)

    print(f"best_index={best_index}")
    print(f"best_fitness={best_fitness},best_predict={predict_values[best_index]}")
    best_solution = populations[best_index]
    # best_atom = Atoms(positions=best_solution, symbols=["Pt"] * len(best_solution))
    last_population = populations
    last_atom_list = []
    for i in range(len(last_population)):
        last_atom_list.append(
            Atoms(
                positions=last_population[i], symbols=["Pt"] * len(last_population[i])
            )
        )
    ini_atom_list = []
    for i in range(len(ini_population_record)):
        ini_atom_list.append(
            Atoms(
                positions=ini_population_record[i],
                symbols=["Pt"] * len(ini_population_record[i]),
            )
        )
    return fitness_record, last_atom_list, ini_atom_list, fitness_values, time_cost_record
    # def genome(num_atoms):
    #     np.randomduplicates exist, then generate a new structure

    # def genome(num_atoms):
    #     np.randomduplicates exist, then generate a new structure
def plot_radar(table,index,descriptors,multiplier,save=False):
    r=[table.iloc[index][descriptors[i]]*multiplier[i] for i in range(len(descriptors))]
    fig=go.Figure(data=go.Scatterpolar(
        r=r,
        theta=[f"{descriptors[i]}*{multiplier[i]}" for i in range(len(descriptors))],
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
        ),
        ),
        showlegend=False
    )
    #fig.show()
    if save:
        fig.write_image(f"radar_{index}.svg",engine="kaleido")
if __name__ == "__main__":
    ini_configurations=toml.load("config.toml")

    fitness_record, last_atom_list, ini_atom_list,fitness_values,time_cost_record = genetic_algorithm(
        max_num_atoms=ini_configurations["max_num_atoms"],
        generations=ini_configurations["generations"],
        population_size=ini_configurations["population_size"],
        lc=ini_configurations["lc"],
        niche_radius=ini_configurations["niche_radius"],
        alpha=ini_configurations["alpha"],
        percentage=ini_configurations["save_config"]["percentage"],
        initial_mutation_rate=ini_configurations["initial_mutation_rate"],
        mutation_rate_decay=ini_configurations["mutation_rate_decay"],
        weight=ini_configurations["fitness_weight"],
        initial_crowding_distance=ini_configurations["initial_crowding_distance"],
        cross_over_rate=ini_configurations["cross_over_rate"],
        elite_fraction=ini_configurations["elite_fraction"],
        fitness_func=ini_configurations["fitness_func"],
        reduction=ini_configurations["reduction"],
        descriptors=ini_configurations["descriptors"],
    )

    particles_descriptors = []
    rankings=ranking_fitness(fitness_values)
    print()
    index=[]
    for i in range(len(rankings)):
        index.extend(np.where(rankings==i+1)[0])
        if len(index)>=len(rankings)*ini_configurations["save_config"]["percentage"]: # 3/4 of the good solutions
            break
    extend = ini_configurations["sample_name"]
    Atoms_3quarter=[last_atom_list[i] for i in index]
    for i in range(len(Atoms_3quarter)):
        particles_descriptors.append(descriptor_table(Atoms_3quarter[i], all=True))
    np.savetxt("fitness_all.txt",fitness_record)
    np.savetxt("fitness_last.txt",fitness_values)
    pdes = pd.DataFrame(particles_descriptors)
    pdes.to_csv("descriptors_" + extend + ".csv")
    pdes[ini_configurations["plot_descriptors"]].hist(bins=100)
    plt.tight_layout()
    plt.savefig("dis_" + extend + ".png")
    if len(Atoms_3quarter)<10:
        for i in range(len(Atoms_3quarter)):
            write(f"best_{i}"+extend+".png",Atoms_3quarter[i],rotation='45x,45y,45z')
            if ini_configurations['plot_radar_conf']['save']:
                plot_radar(pdes,i,ini_configurations['plot_radar_conf']['descriptors'],multiplier=ini_configurations['plot_radar_conf']['multiplier'],save=True)
    else:
        for i in range(10):
            write(f"best_{i}"+extend+".png",Atoms_3quarter[i],rotation='45x,45y,45z')
            if ini_configurations['plot_radar_conf']['save']:
                plot_radar(pdes,i,ini_configurations['plot_radar_conf']['descriptors'],multiplier=ini_configurations['plot_radar_conf']['multiplier'],save=True)
    write_file(Atoms_3quarter, "output_" + extend)
    try:
        draw_ellipsoid(
            Atoms_3quarter[0], save=True, filename="ellipsoid_" + extend + ".png"
        )
    except:
        pass
    f=input("press close to exit") 
