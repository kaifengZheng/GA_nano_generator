# Genetic algorithm for reconstruction of nanoparticles from known descriptors
## The purpose of this project
Decoding the structural and morphological information is crucial after measuring samples containing nanoparticles. This information opens the door to discovering the nano-world. However, experimental tools like spectroscopies typically provide limited information across several variables or descriptors. Constructing a comprehensive view of the sample or representative nanoparticles requires transforming this 1-D information into 3-D, which is an intriguing yet challenging task. This tool offers a method for reconstructing nanoparticles using various known descriptors. The advantage of using a genetic algorithm is that it produces a bunch of nanoparticles rather than a single solution, allowing users to evaluate other computable properties of the generated nanoparticles.
## Methodology
1. Genetic algorithm-a quick introduction
2. Algorithm
   1. particle construction
   2. fitness function
   3. parent selection
   4. mutation
   5. crossover
   6. clowding and niche
3. Configuration
## installation
```bash
git clone https://github.com/kaifengZheng/GA_nano_generator.git
```
## Usage
* Step 1: Modify parameters in config.toml
* Step 2: run NanoGene
``` bash
cd essential_linux
bash NanoGene.py
```



  
