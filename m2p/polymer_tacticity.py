import sys
import m2p
import ast
from m2p import PolyMaker

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as mfs
from rdkit.Chem.Draw import MolsToGridImage as m2g
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem

import pandas as pd
import numpy as np
import shortuuid
import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple
import random
from collections import deque
import re

pm = PolyMaker() #from m2p


def identify_chiral_centers(smiles: str) -> List[int]:
    """
    Identify potential chiral centers in a molecule from SMILES.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        List of tuples where the first number is atom index and second number is distance from first atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Get chiral centers
    chiral_centers = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            chiral_centers.append(atom.GetIdx())
            
    distance_from_start=[]
    for i in chiral_centers:
        dist=get_shortest_distance(smiles, 0, i)
        distance_from_start.append(dist)
    
    centers=list(zip(chiral_centers, distance_from_start))
    
    centers.sort(key=lambda x: x[1])
    
    return centers

def get_shortest_distance(smiles: str, atom1_idx: int, atom2_idx: int) -> int:
    """
    Args:
        smiles: SMILES string
        atom1_idx: Index of first atom
        atom2_idx: Index of second atom
        
    Returns:
        Number of bonds between the two atoms (integer)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    mol = Chem.AddHs(mol)
    
    adj = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a].append(b)
        adj[b].append(a)
    
    visited = {atom1_idx}
    queue = deque([(atom1_idx, 0)])  # (node, distance)
    
    while queue:
        node, dist = queue.popleft()
        
        if node == atom2_idx:
            return dist
        
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # No path found (shouldn't happen for connected molecules)

def isotactic(smiles: str, config: str = 'R') -> str:
    """
    Add isotactic stereochemistry to a polymer SMILES string.
    All chiral centers will be on same side of backbone chain. The first chiral center will have no stereochem due to
    end-chain variability
    Args:
        smiles: Input SMILES string (without stereochemistry)
        config: 'R' for all R configuration, 'S' for all S configuration
        
    Returns:
        SMILES string with isotactic stereochemistry added
    """
    # Parse the SMILES
    centers = identify_chiral_centers(smiles)
    
    if not centers:
        return smiles
    
    mol=Chem.MolFromSmiles(smiles)
    
    base_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if config.upper() == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    alt_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CCW if config.upper() == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CW

    current_tag = base_tag
    
    # Build up tags sequentially based on pairwise distances
    for i in range(1, len(centers)):
        atom = mol.GetAtomWithIdx(centers[i][0])
        distance = get_shortest_distance(smiles, centers[i-1][0], centers[i][0])
        
        # ISOTACTIC
        if distance % 2 == 0:
            atom.SetChiralTag(current_tag)
        else:
            current_tag = alt_tag if current_tag == base_tag else base_tag
            atom.SetChiralTag(current_tag)
            
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def syndiotactic(smiles: str, start_config: str = 'R') -> str:
    """
    Create syndiotactic polymer by considering backbone distances. The first chiral center will have no stereochem due to
    end-chain variability
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        SMILES with proper syndiotactic stereochemistry
    """
    # Parse the SMILES
    centers = identify_chiral_centers(smiles)
    
    if not centers:
        return smiles
    
    mol=Chem.MolFromSmiles(smiles)
    
    # Set alternating stereochemistry
    base_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if start_config.upper() == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    alt_tag = Chem.ChiralType.CHI_TETRAHEDRAL_CCW if start_config.upper() == 'R' else Chem.ChiralType.CHI_TETRAHEDRAL_CW
    
    current_tag = base_tag
    
    for i in range(1, len(centers)): 
        atom = mol.GetAtomWithIdx(centers[i][0])
        distance = get_shortest_distance(smiles, centers[i-1][0], centers[i][0])
        
        # SYNDIOTACTIC
        if distance % 2 == 0:
            current_tag = alt_tag if current_tag == base_tag else base_tag
            atom.SetChiralTag(current_tag)
        else:
            atom.SetChiralTag(current_tag)  
            
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def atactic(smiles: str, r_fraction: float = 0.4, seed: Optional[int] = None) -> str:
    """
    Create atactic polymer with random stereochemistry.
    
    Args:
        smiles: Input SMILES string
        r_fraction: Fraction of stereocenters that should be R
        seed: Random seed for reproducibility
        
    Returns:
        SMILES with atactic stereochemistry
    """
    if seed is not None:
        random.seed(seed)
    
    centers = identify_chiral_centers(smiles)
    
    if not centers:
        return smiles
    
    mol=Chem.MolFromSmiles(smiles)
    
    # Random stereochemistry
    tags = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
    
    for atom_idx, distance in centers:
        atom = mol.GetAtomWithIdx(atom_idx)
        if random.random() < r_fraction:
            atom.SetChiralTag(tags[0])
        else:
            atom.SetChiralTag(tags[1])
    
    mol = Chem.RemoveHs(mol)
    result = Chem.MolToSmiles(mol, isomericSmiles=True)
    
    return result

def add_tacticity(df: pd.DataFrame,
    smiles_column: str = 'smiles_polymer',
    atactic_r_fraction: float = 0.5,
    atactic_seed: Optional[int] = 42,
    verbose: bool = False
) -> pd.DataFrame:
    """
    For polymers with chiral centers:
    - Creates 3 rows: isotactic (pm=1.0), syndiotactic (pm=0.0), atactic (pm=0.5)
    
    For polymers without chiral centers:
    - Keeps original row with pm=0.5
    
    Args:
        df: Input dataframe with polymer SMILES
        smiles_column: Name of column containing polymer SMILES
        atactic_r_fraction: Fraction of R centers for atactic polymers
        atactic_seed: Random seed for atactic generation
        verbose: Print progress information
        
    Returns:
        Expanded dataframe with 'pm' and 'tacticity' columns
    """
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataframe")
    
    new_rows = []
    errors = []
    total = len(df)
    
    rows_list = list(df.iterrows())
    
    for idx, row in rows_list:
        smiles = row[smiles_column]
        
        if verbose and (idx % 100 == 0 or idx == total - 1):
            print(f"Processing row {idx+1}/{total}...")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            chiral_info = identify_chiral_centers(smiles)
            num_chiral = len(chiral_info)
            
            if num_chiral > 0:
                base_row = row.to_dict()
                
                # 1. Isotactic (pm = 1.0)
                row_iso = base_row.copy()
                row_iso[smiles_column] = isotactic(smiles, config='R')
                row_iso['pm'] = 1.0
                row_iso['tacticity'] = 'isotactic'
                new_rows.append(row_iso)
                
                # 2. Syndiotactic (pm = 0.0)
                row_syn = base_row.copy()
                row_syn[smiles_column] = syndiotactic(smiles, start_config='R')
                row_syn['pm'] = 0.0
                row_syn['tacticity'] = 'syndiotactic'
                new_rows.append(row_syn)
                
                # 3. Atactic (pm = 0.5)
                row_atac = base_row.copy()
                seed = atactic_seed + idx if atactic_seed is not None else None
                row_atac[smiles_column] = atactic(smiles, r_fraction=atactic_r_fraction, seed=seed)
                row_atac['pm'] = 0.5
                row_atac['tacticity'] = 'atactic'
                new_rows.append(row_atac)
                
            else:
                # No chiral centers
                row_copy = row.to_dict()
                row_copy['pm'] = 0.5
                row_copy['tacticity'] = 'achiral'
                new_rows.append(row_copy)
                
        except Exception as e:
            errors.append((idx, str(e)))
            if verbose:
                print(f"  Warning: Error processing row {idx}: {e}")
            
            row_copy = row.to_dict()
            row_copy['pm'] = None
            row_copy['tacticity'] = 'error'
            new_rows.append(row_copy)
    
    df_expanded = pd.DataFrame(new_rows)
    
    if verbose:
        print("\n" + "="*80)
        print("EXPANSION SUMMARY")
        print("="*80)
        print(f"Original rows: {len(df)}")
        print(f"Expanded rows: {len(df_expanded)}")
        print(f"\nTacticity distribution:")
        print(df_expanded['tacticity'].value_counts())
        print(f"\npm distribution:")
        print(df_expanded['pm'].value_counts(dropna=False))
        
        if errors:
            print(f"\n⚠ Errors encountered: {len(errors)}")
            for idx, error in errors[:5]:
                print(f"  Row {idx}: {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors)-5} more")
    
    return df_expanded

