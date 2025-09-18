import sys
from ase import Atoms
from ase.visualize import view
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
"""
   Author: Kaifeng Zheng
"""
def ase_plot_more_info(atoms:Atoms,info:np.array):
    """
    plot more information of atoms object

    Args:
        atoms (Atoms): Atoms object
        info (np.array): The additional information of atoms object, which should be 1-d array containing atom-wise information storing atom charge.
    """
    cluster=Atoms(positions=atoms.arrays['positions'], numbers=atoms.arrays['numbers'],charges=np.array(info))
    view(cluster)



def plot_principle_axis(atoms:Atoms,fig,title,plotnumber=111):
    #(2*moment_atom[1]-moment_atom[2]-moment_atom[0])/moment_atom[2]
    pos=atoms.arrays['positions']
    center=np.mean(pos,axis=0)
    pos=pos-center
    pos = np.round(pos,5)
    pca = PCA(n_components=3)
    pos_next=np.round(np.zeros_like(pos))
    max_value=np.max(pos,axis=0)
    min_value=np.min(pos,axis=0)
    c,b,a=np.sort(max_value-min_value)
    c_next,b_next,a_next=0,0,0
    i=0
    while a-a_next>1e-4 or c-c_next>1e-4 or b-b_next>1e-4:
        """
              SCF loop
        """
        pos_next=pos
        max_value=np.max(pos_next,axis=0)
        min_value=np.min(pos_next,axis=0)
        c_next,b_next,a_next=np.round(np.sort(max_value-min_value),5)
        
        result=pca.fit(pos_next)
        pos_fit=result.fit_transform(pos_next)
        pos=np.round(pos_fit,5)
        max_value=np.max(pos,axis=0)
        min_value=np.min(pos,axis=0)
        c,b,a=np.round(np.sort(max_value-min_value),5)
        i+=1
    v=result.components_
    #fig = plt.figure()
    ax = fig.add_subplot(plotnumber,projection='3d')
    
    p=ax.scatter(pos[:,0],pos[:,1],pos[:,2],'.',s=80)
    # a = Arrow3D([center[0], v[0]], [center[1], v[1]], 
    #             [center[2], v[2]], mutation_scale=20, 
    #             lw=3, arrowstyle="-|>", color="r")
    # ax.add_artist(a)
    ax.quiver3D(0,0,0,v[0,0],v[1,0],v[2,0],length=10,color='r',normalize=False)
    ax.quiver3D(0,0,0,v[0,1],v[1,1],v[2,1],length=10,color='r',normalize=False)
    ax.quiver3D(0,0,0,v[0,2],v[1,2],v[2,2],length=10,color='r',normalize=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([np.min(pos[:,0])-1,np.max(pos[:,0])+1])
    ax.set_ylim([np.min(pos[:,1])-1,np.max(pos[:,1])+1])
    ax.set_zlim([np.min(pos[:,2])-1,np.max(pos[:,2])+1])
    ax.set_title(title)
    return v
def plot_2d_pdf(r,gr):
    """
    This function is used to plot the 2d pdf of radial distribution function using the input
    r and gr.
    """
    light = (gr-min(gr))/(max(gr)-min(gr))
    circle = []
    for i in range(0,len(gr)):
        circle.append(plt.Circle((np.max(r),np.max(r)),r[i],color=[0.4*light[i],0.6*light[i],0.6*light[i]],linewidth=3,fill=False))
    
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    
    ax.set_facecolor((0,0,0))
    for i in range(0,len(r)):
        ax.add_artist(circle[i])
    plt.plot(np.max(r),np.max(r),'*',color=[1,1,1])
    plt.xlim([0,np.max(r)*2])
    plt.ylim([0,np.max(r)*2])
    plt.xlabel("r ($\AA$)")
    plt.ylabel("r ($\AA$)")

    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='major',direction='out',labelsize=16,length=4,width=1)
    ax.tick_params(which='minor',direction='out',length=3,width=1)
    ax.tick_params(which='both',right=True, top=True)

    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(2)
    plt.show()


    
