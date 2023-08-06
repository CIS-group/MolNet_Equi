# MolNet_Equi: A Chemically Intuitive, Rotation-Equivariant Graph Neural Network
This is an implementation of our paper "MolNet_Equi: A Chemically Intuitive, Rotation-Equivariant Graph Neural Network":

Jihoo Kim, Yoonho Jeong, Won June Kim, Eok Kyun Lee, Insung S. Choi, [MolNet_Equi: A Chemically Intuitive, Rotation-Equivariant Graph Neural Network] (Chem. Asian J. 2023) (submitted)


## Requirements

* Python 3.7.13
* pyg 2.0.4
* RDKit
* scikit-learn

## Data

* Dipole Moments (From https://figshare.com/articles/dataset/Geometries_and_Dipole_Moments_calculated_by_B3LYP_6-31G_d_p_for_10071_Organic_Molecular_Structures/5716246)
* Freesolv
* ESOL (= delaney)

## Models

The 'data.py' cleans and prepares the dataset for the model training.  
The 'layer.py' and 'model.py' build the model structure.  
The 'main.py' is for training of the model.  
