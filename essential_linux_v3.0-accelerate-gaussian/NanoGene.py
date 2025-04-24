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
from joblib import Parallel, delayed
from hashlib import sha1
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

# generate random lattice with random n1,n2,n3 of the superlattice parimeters, and random numbers of atoms in the lattice.
def particles_encode_gen(lc, num_atoms=None, max_num_atoms=None,record_lattice=[]):
    if num_atoms == None:
        # assert max_num_atoms!=None,"Please provide the maximum number of atoms in the lattice."
        num_atoms = random.randint(12, max_num_atoms)
    num_lattice = max_num_atoms//2 #// 2
    # min_num_points=num_atoms

    # print(f"length={np.mean(lattice_big,axis=0)}")
    # times=random.randint(1,3)
    # the extreme case is that the particle has only one layer of unit cells, which has the dimension: np.sqrt(n_max**3)*np.sqrt(n_max**3)*1
    n1 = random.randint(2, int(np.sqrt(num_lattice)))
    n2 = random.randint(2, int(np.sqrt(num_lattice)))
    n3 = random.randint(1, int(np.sqrt(num_lattice)))
    # n3 = random.randint(1, int(np.sqrt(num_lattice)))
    # n1 = random.randint(n3, int(np.sqrt(num_lattice)))
    # n2=n1
    
    lattice = empty_lattice(lc, n1, n2, n3)
    num_points = len(lattice)
    # this is too strong
    while num_points <= num_atoms or [n1,n2,n3,num_atoms] in record_lattice:
        if num_atoms == None:
            # assert max_num_atoms!=None,"Please provide the maximum number of atoms in the lattice."
            num_atoms = random.randint(12, max_num_atoms)
        n1 = random.randint(2, int(np.sqrt(num_lattice)))
        n2 = random.randint(2, int(np.sqrt(num_lattice)))
        n3 = random.randint(1, int(np.sqrt(num_lattice)))
        # n3 = random.randint(1, int(np.sqrt(num_lattice)))
        # n1 = random.randint(n3, int(np.sqrt(num_lattice)))
        # n2=n1
        lattice = empty_lattice(lc, n1, n2, n3)
        num_points = len(lattice)
    record_lattice.append([n1,n2,n3,num_atoms])
    # print(record_lattice)
    lattice = lattice[:num_atoms]

    return lattice,record_lattice



def ini_population(
    lc, population_size, num_atoms=None, max_num_atoms=None,lc_record=[]
):
    generations = []
    n = 0
    for i in range(population_size):
        particle,lc_record = particles_encode_gen(
            lc,
            num_atoms=num_atoms,
            max_num_atoms=max_num_atoms,
            record_lattice=lc_record #record lattice numbers to avoid duplicates
        )
        # print(nn)
        generations.append(particle)
    return generations,lc_record


def fitness(particle, descriptors: dict, fitness_func="L1", reduction="sum",weight:np.ndarray=None,soft_penalty=False):
    if weight is None:
        weight=np.array([1]*len(descriptors.keys()))
    if len(weight)<len(descriptors.keys()):
        raise ValueError("Number of weights is less then the number of descriptors!")
    if len(weight)>len(descriptors.keys()):
        raise ValueError("Number of weights is more then the number of descriptors!")
    atom = Atoms(positions=particle, symbols=["Pt"] * len(particle))
    try:
            cluster_descriptors = descriptor_table(
                atom,all=False,descriptors=list(descriptors.keys())
            )
    except RuntimeWarning:
        print(f"calculate descriptor: {atom.get_positions()}")
    true_dis = []
    tolerances=[]
    for k in descriptors.keys():
        if k != "ellipsoid":
            true_dis.append(descriptors[k][0])
            tolerances.append(descriptors[k][1]) # tolerance for L1 and L2 distance, this is used for soft penalty in the similarity calculation
        else:
            true_dis.append(descriptors[k][0][0])
            true_dis.append(descriptors[k][1][0])
            true_dis.append(descriptors[k][2][0])
            tolerances.append(descriptors[k][0][1]) 
            tolerances.append(descriptors[k][1][1]) 
            tolerances.append(descriptors[k][2][1])
    true_dis = np.array(true_dis)
    tolerances = np.array(tolerances)
    # print(true_dis)
    # if "oblateness_moment" not in descriptors.keys():
    #     pred_dis = np.array(
    #         [cluster_descriptors[k] for k in list(cluster_descriptors.keys())[:-1]]
    #     )
    # else:
    pred_dis=np.array([cluster_descriptors[k] for k in list(cluster_descriptors.keys())])
    # print(pred_dis)
    # MSE
    # np.sum((true_dis-pred_dis)**2)
    sim = 0
    if soft_penalty:
        sim= 1-np.exp(-((true_dis - pred_dis) ** 2) / (2 * (tolerances ** 2)))
    else:
        diff=np.abs(true_dis - pred_dis)
        sim = np.where(diff <= tolerances, 0,100*diff)  # soft penalty for L1 and L2 for L1 and L2
        if fitness_func=="L1":
            sim = np.abs(sim)
        elif fitness_func=="L2":
            sim = np.power(sim, 2)
        elif fitness_func=="RAE":
            sim = sim / true_dis

    if reduction=="sum":
        sim = np.dot(sim,weight) 
    elif reduction=="mean":
        # print(f"mean={sim}")
        sim = np.dot(sim,weight)/np.sum(weight) 
    elif reduction=="max":
        sim = np.max(np.multiply(sim,weight)) 

    return sim, pred_dis


# def parents(population, fitness_values, tol=1e5):
#     fitness_values = np.array(fitness_values)
#     ranking = ranking_fitness(fitness_values)
#     prob = normalized_p(ranking)
#     # index=np.arange(len(population))
#     # threshold=random.uniform(np.min(prob),np.max(prob))
#     current = 0
#     index = 0
#     # for i in range(len(population)):
#     while index < tol:
#         i = random.randint(0, len(population) - 1)
#         index += 1
#         if random.uniform(0, 1) <= prob[i]:
#             return list(population[i])
#         if index == tol:
#             raise Exception("exceed the maximum try times!")
def parents(population, fitness_values,k=5, minimization=True):
    parent1 = tournament_selection(population, fitness_values, k=k, minimization=minimization)
    return parent1
def tournament_selection(population, fitness, k=3, minimization=True):
    indices = np.random.choice(len(population), k, replace=False)
    selected = [population[i] for i in indices]
    selected_fitness = [fitness[i] for i in indices]

    if minimization:
        winner_index = np.argmin(selected_fitness)
    else:
        winner_index = np.argmax(selected_fitness)
    
    return selected[winner_index]


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
    lamb = np.log(10) / (len(fitness_ranking) - 1)
    # print(lamb)
    z = np.array([np.exp(lamb * r) for r in fitness_ranking])
    z_sum = np.sum(z)
    return z / z_sum



def normalized_p(fitness_ranking):
    p = probability_selection(fitness_ranking)
    return p


def atom_cut_up_down_center(particle, theta, phi,center):
    def normal_vector(center,theta, phi):
        plane_normal=np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        plane_normal=plane_normal/np.linalg.norm(plane_normal)
        return plane_normal

    num_atoms = len(particle)
    particle = np.array(particle)
    if num_atoms < 5:
        print("Warning particle is less than 5 atoms")
        return particle,[]

    particle = np.array(particle)
    normal = normal_vector(center,theta, phi)
    trancate_upparticle = []
    trancate_downparticle = []
    distances=np.dot(particle-center,normal)
    trancate_upparticle=particle[distances>0]
    trancate_downparticle=particle[distances<=0]
    return np.array(trancate_upparticle), np.array(trancate_downparticle)

def align_crystal(particle, lattice_big):
    select=np.random.randint(len(particle))
    center_A=particle[select]
    distance=cdist([center_A],lattice_big)[0]
    index=np.where(distance==np.min(distance))[0][0]
    center_B=lattice_big[index]
    diff=center_A-center_B    
    particle=particle-diff
    return particle

def remove_duplicates(particle, lc):
        duplicates = []
        index_get = []
        for i in range(len(particle)):
            dis=cdist([particle[i]], particle)[0]
            dis_sort=sorted(dis)
            
            if np.isclose(dis_sort[1],0,rtol=1e-5):
                  index=np.where(dis==dis_sort[1])[0][0]
                  index_get.append(sorted([i,index]))
        for i in range(len(index_get)):
               duplicates.append(index_get[i][1])
        
        duplicates=np.unique(duplicates)
        return np.array([particle[i] for i in range(len(particle)) if i not in duplicates])

def crossover(lc, parent1, parent2, cross_over_rate=0.6,center_method="mean"):
    parent2=align_point_clouds_pca(parent2, parent1)
    if random.random() < cross_over_rate:
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        min_dim=np.min(parent1,axis=0)
        max_dim=np.max(parent1,axis=0)
        diff=np.min(max_dim-min_dim)/2
        center_real = np.mean(parent1, axis=0)
        min_dim=np.min(parent2,axis=0)
        max_dim=np.max(parent2,axis=0)
        diff=np.min(max_dim-min_dim)/2
        center_real2 = np.mean(parent2, axis=0)
        center_diff=center_real-center_real2
        parent2=parent2+center_diff
        
        if center_method=="random":
            center0=np.random.uniform(center_real[0]-np.sqrt((diff**2)/3),center_real[0]+np.sqrt((diff**2)/3))
            center1=np.random.uniform(center_real[1]-np.sqrt((diff**2)/3),center_real[1]+np.sqrt((diff**2)/3))
            center2=np.random.uniform(center_real[2]-np.sqrt((diff**2)/3),center_real[2]+np.sqrt((diff**2)/3))
        elif center_method=="mean":
            center0=center_real[0]
            center1=center_real[1]
            center2=center_real[2]
        else:
            raise ValueError(f"\"{center_method}\" can not be reconized as method, center method should be either \"random\" or \"mean\" (default: center_method=\"center\")!")
        center=np.array([center0,center1,center2])
        # particle_update1,particle_update2  = atom_cut_up_down_center(particle, theta, phi,center)
        # parent1_particle=recon_coordfromcodes(parent1,lc,max_num_atoms)
        # parent2_particle=recon_coordfromcodes(parent2,lc,max_num_atoms)
        try:
            parent1_up, parent1_down = atom_cut_up_down_center(parent1, theta, phi, lc,center)
            parent2_up, parent2_down = atom_cut_up_down_center(parent2, theta, phi, lc,center)
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

def shift_particles(particle1,particle2,lc):
        dist=cdist(particle1,particle2)
        short=np.min(dist,axis=1)
        index_A=np.where(short==np.min(short))[0][0]
        index_B=np.where(dist[index_A]==np.min(short))[0][0]
        # print(f"short={np.min(dist)}")
        
        if np.min(dist)<=lc/2:
               shift_A=particle1
        else:
               diffB_to_A=particle2[index_B]-particle1[index_A]
               shift_A=particle1+diffB_to_A
        return shift_A,particle2

def mutation(particle, mutation_rate=0.3,center_method="mean"): #center_method="mean"/"random"
    mr = random.uniform(0, 1)
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

    if mr <= mutation_rate * 1 / 2 and len(particle)>13:  # and mr <= mutation_rate * 2 / 3:
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2 * np.pi)
        min_dim=np.min(particle,axis=0)
        max_dim=np.max(particle,axis=0)
        diff=np.min(max_dim-min_dim)/2
        center_real = np.mean(particle, axis=0)
        if center_method=="random":
            center0=np.random.uniform(center_real[0]-np.sqrt((diff**2)/3),center_real[0]+np.sqrt((diff**2)/3))
            center1=np.random.uniform(center_real[1]-np.sqrt((diff**2)/3),center_real[1]+np.sqrt((diff**2)/3))
            center2=np.random.uniform(center_real[2]-np.sqrt((diff**2)/3),center_real[2]+np.sqrt((diff**2)/3))
        elif center_method=="mean":
            center0=center_real[0]
            center1=center_real[1]
            center2=center_real[2]
        else:
            raise ValueError(f"\"{center_method}\" can not be reconized as method, center method should be either \"random\" or \"mean\" (default: center_method=\"center\")!")
        center=np.array([center0,center1,center2])
        particle_update1,particle_update2  = atom_cut_up_down_center(particle, theta, phi,center)
        if len(particle_update1)>13 and len(particle_update2)>13:
            if len(particle_update1)<=len(particle_update2):
                particle_update = particle_update2
            elif len(particle_update1)>len(particle_update2):
                particle_update = particle_update1
        else:
            particle_update = particle
    elif mr >= mutation_rate * 1 / 2 and mr<mutation_rate and len(particle)>13:
        # 2. initialization of populations
        planes=[111,100,110]
        plane=random.choice(planes) # randomly select a plane to shift the particles, this is to avoid bias in the mutation process.
        partcile_updata=cut_by_surface(particle,plane,1)
        int_num=np.random.randint(1, len(particle)-13+1) # number of atoms to add/remove, at least 1 atom.
        idx=np.random.choice(len(particle),int_num)
        particle_update = np.delete(particle, idx, axis=0)  
        
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



def find_most_similar(population, individual):
    return np.array(
        [
            min(
                population,
                key=lambda x: calculate_similarity(x, individual),
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

    # print(particle1.shape)
    # print(particle2.shape)
    source_centered = particle1 - np.mean(particle1, axis=0)
    target_centered = particle2 - np.mean(particle2, axis=0)

    # Perform PCA
    if is3Ddimension(source_centered) and is3Ddimension(target_centered):
        try:
            source_pca = PCA(n_components=3).fit(source_centered)
        except RuntimeWarning:
            print(f"source particle dimension is less than 3")
        try:
            target_pca = PCA(n_components=3).fit(target_centered)
        except RuntimeWarning:
            print(f"source particle dimension is less than 3")
    else:
        return particle1
        # Align principal components
    
    
    R = np.dot(target_pca.components_.T, source_pca.components_)
    
    aligned_source = np.dot(source_centered, R.T) + np.mean(particle2, axis=0)

    return aligned_source




def sharing_function(ind1, ind2, niche_radius):
    distance = calculate_similarity(ind1, ind2)
    return (1-(distance+1e-8)/niche_radius) if distance < niche_radius else 1e-8

def calculate_similarity(ind1, ind2):
    # atoms1 = Atoms(positions=ind1, symbols=["Pt"] * len(ind1))
    # atoms2 = Atoms(positions=in2, symbols=["Pt"] * len(in2))
    # des_atoms1 = descriptor_table(atoms1)
    # score1 = np.array(np.array([des_atoms1[k] for k in des_atoms1.keys()]))
    # des_atoms2 = descriptor_table(atoms2)
    # score2 = np.array([des_atoms2[k] for k in des_atoms2.keys()])
    # return np.mean((score1 - score2) ** 2)
    ind1 = align_point_clouds_pca(ind1, ind2)
    dist = chamfer_distance(ind1, ind2)
    # print(dist)

    return dist 


def shared_fitness_value(individual, descriptor, population, niche_radius, fitness_func,reduction,weight,soft_penalty=False,tolerances=1e-5):
    fitnessvalue, predict_value = fitness(individual, descriptor,fitness_func,reduction,weight, soft_penalty=soft_penalty, tolerances=tolerances)
    sharing_sum = np.sum(
        [sharing_function(individual, other, niche_radius) for other in population if not np.array_equal(individual, other)] # avoid self-comparison
    )
    # print(sharing_sum)
    return fitnessvalue *(1+sharing_sum), predict_value


def fitness_sharing(population, descriptor, niche_radius, fitness_func="L1",share=False,reduction="sum",weight=None,soft_penalty=False):
    
    shared_fitness = []
    predict_values = []
    distance_cache={}
    fitness_results = Parallel(n_jobs=-1)(
    delayed(fitness)(individual, descriptor,fitness_func,reduction,weight, soft_penalty=soft_penalty)
    for individual in population
    )
    fitness_value, predict_values = zip(*fitness_results)
    if not share:
        # If sharing is not enabled, return the original fitness values
        return list(fitness_value), list(predict_values)
    
    for i,individual in enumerate(population):
        sharing_sum=0
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            key=tuple(sorted((i,j)))
            if key not in distance_cache:
                # Calculate the similarity only once for each pair of individuals
                distance_cache[key] = calculate_similarity(individual, ind2)
            s_ij=sharing_function(individual, ind2, niche_radius)
            sharing_sum += s_ij
        shared_fitness.append(
            fitness_value[i]*(1+sharing_sum))
          
    # print(max(shared_fitness))
    return shared_fitness, predict_values

def crowding(parent1,parent2,child1,child2,descriptors,fitness_func,reduction,weight=None, soft_penalty=False, tolerances=1e-5):
    # print(f"parent1={parent1}")
    # print(f"parent2={parent2}")
    # print(f"child1={child1}")
    # print(f"child2={child2}")
    # print([len(parent1),len(parent2),len(child1),len(child2)])
    pc11 = calculate_similarity(parent1, child1) 
    pc22 = calculate_similarity(parent2, child2)
    pc12 = calculate_similarity(parent1, child2)
    pc21 = calculate_similarity(parent2, child1)

    fitp1, _ = fitness(parent1, descriptors,fitness_func,reduction,weight, soft_penalty=soft_penalty)
    fitp2, _ = fitness(parent2, descriptors,fitness_func,reduction,weight, soft_penalty=soft_penalty)
    fitch1, _ = fitness(child1, descriptors,fitness_func,reduction,weight, soft_penalty=soft_penalty)
    fitch2, _ = fitness(child2, descriptors,fitness_func,reduction,weight, soft_penalty=soft_penalty)
    if pc11+pc22 <= pc12+pc21:
        new_parent1 = child1 if fitch1 < fitp1 else parent1
        new_parent2 = child2 if fitch2 < fitp2 else parent2
    else:
        new_parent1 = child2 if fitch2 < fitp1 else parent1
        new_parent2 = child1 if fitch1 < fitp2 else parent2
    return new_parent1,new_parent2

def write_file(Atoms_list, foldername):
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    n = 1
    for atoms in Atoms_list:
        write(f"{foldername}/individual_{n}.xyz", atoms, format="xyz")
        n += 1
def find_index_2d(array_2d, array_search):
    """
    Finds the index of a 1D array within a 2D NumPy array.

    Args:
        Array_2d: A list of 2D-array.
        array_search: A specific array to search for.

    Returns:
        The index of the first occurrence of array_1d in array_2d, or -1 if not found.
    """
    for i, row in enumerate(array_2d):
        if np.array_equal(row, array_search):
            return i
    return -1

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
def hash_particle(particle):
    return sha1(np.round(particle,4).tobytes()).hexdigest() # use sha1 to hash the particle, this will help to avoid duplicates in the population, round to 4 decimal places to avoid floating point issues.


def genetic_algorithm(
    max_num_atoms=200,
    generations=40,
    population_size=100,
    selection_pressure=5,
    lc=3.924,
    cross_over_rate=0.3,
    elite_num=2,
    niche_radius=0.1,
    initial_mutation_rate=1,
    mutation_rate_decay=0.9,
    weight=None,
    share=False,
    fitness_func="L1",
    reduction="sum",
    soft_penalty=False,
    descriptors={"oblateness_moment": (1,0.01), "atom_number": (55,5)},
    patience=10,
    early_stopping_threshold=0.01,
    # "CN1":6,
    # "CN2":1.5,
    # "CN3":2,
    # "CN4":5
):

    # 1. set up parameters

    print(descriptors)
    num_lattice = max_num_atoms  # since one unit cell of fcc lattice has 2 atoms
    n_max = int(np.ceil(np.cbrt(num_lattice)))
    lattice_big = extendfcc(
        fccbasis(lc),
        lc,
        int(np.sqrt(n_max**3)),
        int(np.sqrt(n_max**3)),
        int(np.sqrt(n_max**3)),
    )
    lattice_big = lattice_big - np.mean(lattice_big, axis=0)
    lc_record=[]
    # 2. initialization of populations
    populations ,lc_record= ini_population(
        lc=lc,
        population_size=population_size,
        num_atoms=None,
        max_num_atoms=max_num_atoms,
        lc_record=lc_record
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
    fitness_values = fitness_sharing(population=populations, descriptor=descriptors, 
                                     niche_radius=niche_radius,
                                     fitness_func=fitness_func,share=share,reduction=reduction,weight=weight,soft_penalty=soft_penalty)[0]
    des_keys=list(descriptors.keys())
    print(f"initial population={len(populations)}")
    print_descriptors=[{des_keys[i]:descriptors[des_keys[i]][0]} for i in range(len(des_keys))] # for print out the descriptors used in the optimization
    print(f"fit_property={print_descriptors}")
    if share==False:
        print(f"fitness_func={fitness_func}   reduction={reduction}   weight={weight}")
    else:
        print(f"fitness_func={fitness_func}_share_on   reduction={reduction}   weight={weight}")
    stopping=0
    for generation in (pbar := tqdm(range(generations))):
        time_start=time()
        # clustering_energy_values=[clustering_energy(genome,lattice) for genome in population]
        # (1). generate new populations
        fit_pop = dict()
        seen_hashes=set()
        for i in range(len(populations)):
            fit_pop[fitness_values[i]] = populations[i]
        sorted_population = [
            sorted(fit_pop.items(), key=lambda item: item[0], reverse=False)[i][1]
            for i in range(len(fit_pop))
        ]
        # print(population_size,elite_fraction)
        elites = sorted_population[:elite_num]
        populations_new = []  # new population for the next generation
        elites_index=np.unique(np.array([find_index_2d(populations,elites[i]) for i in range(len(elites))]))
        for p in elites:
            h=hash_particle(p)
            if h not in seen_hashes:
                seen_hashes.add(h)
                populations_new.append(p)
        population_residual=[populations[i] for i in range(len(populations)) if i not in elites_index]
        num_steps=len(populations) # number of steps for each generation, each step will produce 2 children
        fitness_values_residual = [fitness_values[i] for i in range(len(populations)) if i not in elites_index] # fitness values for the residual population, this will be used for the tournament selection of parents
        for i in (pbar:=tqdm(range(num_steps))):
            # [1]. find parents from populations based on fitness values
            # print(len(populations),len(parameters),len(fitness_values))
            parent1 = parents(population_residual, fitness_values_residual,k=selection_pressure,minimization=True) # tournament selection for parents
            parent2 = parents(population_residual, fitness_values_residual,k=selection_pressure,minimization=True)
            children1, children2 = crossover(
                lc, parent1, parent2,cross_over_rate=cross_over_rate,center_method="random"
            )

            # print(f"children={len(np.where(np.array(children1)==1)[0]),len(np.where(np.array(children2)==1)[0])}")
            children1 = mutation(
                children1, mutation_rate=mutation_rate,center_method="random"
            )
            children2 = mutation(
                children2, mutation_rate=mutation_rate,center_method="random"
            )
            # convex hull

            children1 = convex_particle(children1, lattice_big, lc)
            children2 = convex_particle(children2, lattice_big, lc)

            # print(len(children1),len(children2))
            #children1, children2 = crowding(parent1=parent1,parent2=parent2,
                                                #child1=children1,child2=children2,
                                                #descriptors=descriptors,
                                                #fitness_func=fitness_func,reduction=reduction,
                                                #weight=weight,soft_penalty=soft_penalty)
            h1=hash_particle(children1)
            h2=hash_particle(children2)
            if h1 not in seen_hashes:
                seen_hashes.add(h1)
                populations_new.append(children1)
            if h2 not in seen_hashes:
                seen_hashes.add(h2)
                populations_new.append(children2)
            if len(populations_new) >= population_size:
                break
        print(f"populations_new={len(populations_new)}")
        if len(populations_new)>population_size:
            populations_new=populations_new[:population_size]
        if len(populations_new)<population_size:
            populations_new.extend(ini_population(
            lc=lc,
            population_size=population_size-len(populations_new), # fill the population to the original size
            # lattice_big=lattice_big,
            max_num_atoms=max_num_atoms,
            num_atoms=None,
        )[0])
        # (2). calculate fitness for new populations and obtain the predicted descriptors
        fitness_values = []
        predict_values = []
        fitness_values, predict_values = fitness_sharing(
            population=populations_new, descriptor=descriptors, niche_radius=niche_radius, 
            fitness_func=fitness_func,share=share,reduction=reduction,weight=weight, soft_penalty=soft_penalty
        )
        fitness_v, _ = fitness_sharing(
            populations, descriptors, niche_radius=0.01,weight=[1]*len(descriptors),fitness_func="L1",share=False,soft_penalty=False,reduction=reduction
        )
        best_fitness = min(fitness_v)
        ave_fitness = np.mean(fitness_v)
        # std_fitness = np.std(fitness_values)
        # best_predict = predict_values[fitness_values.index(best_fitness)]
        fitness_value_sort = sorted(fitness_v)
        quater_fitness_index = np.where(
            np.array(fitness_v)
            <= fitness_value_sort[int(len(fitness_v) * 0.75)]
        )[0][0]
        predict_quater = predict_values[quater_fitness_index]

        best_index = fitness_values.index(best_fitness)
        fitness_record.append(ave_fitness)  # dong
        mutation_rate *= mutation_rate_decay
        time_end = time()
        time_cost = time_end - time_start
        time_cost_record.append(time_cost)
        # print(mutation_rate)
        # [4]. print output
        print(
            f"Generation{generation}: Finess={np.round(np.mean(fitness_v),3)}+/-{np.round(np.std(fitness_v),3)} Predict_Q3={predict_quater} mutation_rate={np.round(mutation_rate,5)} time_cost={time_cost}"
        )

        # [6]. early stopping
        if len(fitness_record)<patience:
            continue
        else:
            recent=fitness_record[-1]
            if fitness_record[-2]-fitness_record[-1]<early_stopping_threshold:
                stopping+=1
                print(f"stopping={stopping}")
            if stopping >= patience:
                break
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
        selection_pressure=ini_configurations["selection_pressure"],
        share=ini_configurations["share"],
        soft_panelty=ini_configurations["soft_penalty"], # using gaussian-like loss function
        patience=ini_configurations["patience"],
        ealy_stopping_threshold=ini_configurations["early_stopping_threshold"],
        lc=ini_configurations["lc"],
        niche_radius=ini_configurations["niche_radius"],
        percentage=ini_configurations["save_config"]["percentage"],
        initial_mutation_rate=ini_configurations["initial_mutation_rate"],
        mutation_rate_decay=ini_configurations["mutation_rate_decay"],
        weight=ini_configurations["fitness_weight"],
        cross_over_rate=ini_configurations["cross_over_rate"],
        elite_num=ini_configurations["elite_num"],
        fitness_func=ini_configurations["fitness_func"],
        reduction=ini_configurations["reduction"],
        descriptors=ini_configurations["descriptors"])
    

    particles_descriptors = []
    rankings=ranking_fitness(fitness_values)
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
