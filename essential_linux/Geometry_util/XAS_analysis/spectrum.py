import numpy as np
from scipy.interpolate import interp1d

def energy_grids_inter_multi(ref,spectrum):
    """
    regrid spectra to the energy grid of reference spectrum

    Args:
        ref (np.array): 2-d array,reference spectrum, which is the first calculated spectrum, first column is energy, 
        second column is intensity spectrum (np.array): 3-d array, first column is samples, second column is energy. 
        The third dimension is intensity. 
        e.g:
                array([[[ 3,  2,  3],
                        [ 4,  5,  6]],

                    [[ 7,  8,  9],
                        [10, 11, 12]]])
                
    """
    min_spe = np.max([spectrum[i][:,0][0] for i in range(len(spectrum))])
    max_spe = np.min([spectrum[i][:,0][-1] for i in range(len(spectrum))])
    emesharray = ref[:,0]
    cond1 = emesharray[emesharray<max_spe]
    cond2 = cond1[cond1>min_spe]
    emesh = cond2
    
    for i in range(len(spectrum)):
        #FDMNES
        #FEFF
        #print(spectrumAB[2][0])
        meshspectrumX = interp1d(spectrum[i][:,0],spectrum[i][:,1])
        ##remesh spectrum
        meshspectrumY = meshspectrumX(emesh)
        spectrum[i]=np.array([emesh,meshspectrumY])
        

# def
#     #Energy aligned for experimental spectrum    
#     expspectrumX = interp1d(foilexperiment[0],foilexperiment[1])
#     expspectrumY = expspectrumX(emesh)
#     expspectrum = np.array([emesh,expspectrumY])
        