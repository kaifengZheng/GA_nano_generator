# Genetic algorithm for reconstruction of nanoparticles from known descriptors
<p align="center"><img width="551" height="132" alt="image" src="https://github.com/user-attachments/assets/44eb738c-d678-4f69-943e-ca3e024022c2" />

The purpose of the code
------------------------
* Reconstruct a bunch of possible nanoparticle structures from given structural or geometrical descriptors taking advantage of genetic algorithm. 🔥
* Those reconstructed structures can be employed to further validate experimental observation, to build a structure database for machine learning studies or other purposes.  ⚡
* This prototype method only works for FCC particles for this moment, other lattice types can be 

Methodology
------------------------
The basic method is illustrated below, which follows a standard genetic algorithm framework.
<p align="center"><img width="701" height="758" alt="image" src="https://github.com/user-attachments/assets/16315b0c-3fb9-4576-867c-76af5fd58d13" />
  
<br>The available descriptors are listed below:
<p align="center"><img width="763" height="446" alt="image" src="https://github.com/user-attachments/assets/73039607-5d18-4def-a3c5-4b34cf555511" />

Particle disstribution
------------------------
The selected descriptors can be recompiled from reconstructed structures. As the following figure demonstrated, those particles will have board distributions for many descriptors and converged to a narrow distribution for the given value.<br>
<p align="center"><img width="990" height="467" alt="image" src="https://github.com/user-attachments/assets/3b7dd58f-eb0e-4aea-9ff4-a878f492fa2a" />

Installation
-------------------------
```bash
git clone https://github.com/kaifengZheng/GA_nano_generator.git
```

Usage
-------------------------
```bash
cd essential_linux
python NanoGene.py
```

Citation
--------------------------
please cite: <br>
Kaifeng Zheng, Charlotte Vogt, Anatoly I. Frenkel; Morphological descriptors of nanoparticles: The link between atomistic structures and x-ray absorption spectra. J. Chem. Phys. 28 January 2026; 164 (4): 044201. https://doi.org/10.1063/5.0301368




  
