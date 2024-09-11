from ase.io import read 
import numpy as np 
import jax
import jax.numpy as jnp 
from scipy.optimize import minimize
import pickle
import os

from matscipy.neighbours import neighbour_list 
from datetime import datetime
import jraph

import importlib.util

def import_base_model(model_select):
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'model.py')
    if os.path.exists(model_path):
        spec = importlib.util.spec_from_file_location("model", model_path)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        print("Using local model import")
        return model.GraphNN
    else:
        if model_select == 'nequip':
            from bam_mol.model.nequip_model import GraphNN
            print("Using nequip model import")
            return GraphNN

        else:
            raise ValueError(f"Unknown model selection: {model_select}")

def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y

# from jraph.examples
def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
        7 nodes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs

    Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
        A graphs_tuple batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    #pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    # since the node size (the number of atoms) is fixed
    pad_nodes_to = jnp.sum(graphs_tuple.n_node) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

def get_graphset (data, cutoff, nbatch, uniq_element, enr_avg_per_element ):
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    graph_list = []
    for atoms in data:
        crds = atoms.get_positions()
        
        node_enr_avg = jnp.array([enr_avg_per_element[uniq_element[iz]] 
                                  for iz in atoms.numbers])
        
        enr = atoms.get_potential_energy() - node_enr_avg.sum() 
        
        iatoms, jatoms, Sij = neighbour_list (quantities='ijS',
                                              atoms=atoms,
                                              cutoff=cutoff,
                                              )

        species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
        num_nodes = crds.shape[0]
        num_edges = iatoms.shape[0]

        # positions, species, forces depend on nodes
        # energy, cell, volume, stress depend on each configuration (globals)

        if 'forces' in atoms._calc.results.keys():
            frc = atoms.get_forces ()

            graph = jraph.GraphsTuple (
                nodes={
                    "positions":jnp.array (crds),
                    "species":species,
                    "forces" : jnp.array(frc)
                },
                edges=dict(Sij=jnp.array(Sij)),
                globals=dict(energy=jnp.array([enr]),
                             forces=jnp.array(frc)),
                ###
                senders=jnp.array(iatoms),
                receivers=jnp.array(jatoms),
                n_node=jnp.array([num_nodes]),
                n_edge=jnp.array([num_edges]),
            )
        else:
            graph = jraph.GraphsTuple (
                nodes={
                    "positions":jnp.array (crds),
                    "species":species,
                },
                edges=dict(Sij=jnp.array(Sij)),
                globals=dict(energy=jnp.array([enr])),
                ###
                senders=jnp.array(iatoms),
                receivers=jnp.array(jatoms),
                n_node=jnp.array([num_nodes]),
                n_edge=jnp.array([num_edges]),
            )

        graph_list.append (graph)

    pad_nodes_to = 0 #nbatch*max_nodes+1
    pad_edges_to = 0 #nbatch*max_edges
    n_data = len(data)
    for ist0 in range (0, n_data, nbatch):
        ied0 = jnp.where (ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch (graph_list[ist0:ied0])
        pad_nodes_to = max(graph.n_node.sum(), pad_nodes_to)
        pad_edges_to = max(graph.n_edge.sum(), pad_edges_to)

    pad_nodes_to = pad_nodes_to + 1
    pad_graphs_to = nbatch + 1
    dataset_list = []
    for ist0 in range (0, n_data, nbatch):
        ied0 = jnp.where (ist0+nbatch < n_data, ist0+nbatch, n_data)
        graph = jraph.batch (graph_list[ist0:ied0])
        graph = jraph.pad_with_graphs(graph,
                                      pad_nodes_to,
                                      pad_edges_to,
                                      pad_graphs_to)
        dataset_list.append (graph)

    return dataset_list


def get_enr_avg_per_element (traj, element):

    tgt_enr = np.array([atoms.get_potential_energy() 
                    for atoms in traj])
    
    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([ (atoms.numbers == e).sum()
                                   for atoms in traj])
                                for e, i in uniq_element.items()}
    c0 = jnp.array ([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum()/c0.sum()
    w0 = jnp.array ([m0 for _ in element])
    
    def loss_fn (weight, count):
        # weight:  (nspec)
        # count:  (nspec, ndata)
        
        def objective_mean (w0, c0): 
            # w0: weight (nspec)
            # c0: count  (nspec, ndata)
            return np.einsum('i,ij->j', w0, c0)
    
        prd_enr = objective_mean (weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff*diff).mean()
    
    results = minimize (loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x
    
    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]
    
    return enr_avg_per_element, uniq_element



def get_trajectory (fname, ntrain, nvalid, cutoff, nbatch, element, rng):
    if type(ntrain) == str:
        train_data = read (ntrain, index=slice(None))
        valid_data = read (nvalid, index=slice(None))
        print(f'\nntrain: {len(train_data)} | nvalid: {len(valid_data)}\n')
        traj = train_data + valid_data
    else:
        nsamp = ntrain + nvalid
        traj = read (fname, index=slice(None))[-nsamp:]
        idx = jnp.arange(nsamp)
        
        idx = jax.random.permutation (rng, idx)
        idx_train = idx[:ntrain]
        idx_valid = idx[ntrain:]
        train_data = [traj[i] for i in idx_train]
        valid_data = [traj[i] for i in idx_valid]

    enr_avg_per_element, uniq_element = \
        get_enr_avg_per_element(traj, element)
    print ('enr_avg_per_element', enr_avg_per_element)

    train_graphset = get_graphset (train_data, cutoff, nbatch, 
                                   uniq_element, enr_avg_per_element)
    valid_graphset = get_graphset (valid_data, cutoff, nbatch, 
                                   uniq_element, enr_avg_per_element)
                                  
    return train_graphset, valid_graphset, uniq_element, enr_avg_per_element


def get_trajectory_test(fname, ndata, cutoff, nbatch, 
                        model_ckpt): #enr_ave, enr_std):
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    traj_test = read(fname, index=slice(None))[-ndata:]
    
    uniq_element = model_ckpt['uniq_element']
    enr_avg_per_element = model_ckpt['enr_avg_per_element']

    graphset = get_graphset (traj_test, cutoff, nbatch, 
                             uniq_element, enr_avg_per_element)
    return graphset



def get_graphset_to_predict_from_atoms(atoms, cutoff, 
                                       uniq_element):
    # iatoms ==> sender
    # jatoms ==> receiver
    # Dij = R[jatom] - R[iatom] + jnp.einsum('ei,ij->ej', Sij, cell)
    # Dij = R[jatom] - R[iatom] + Sij.dot(cell)
    crds = atoms.get_positions()
    cell = atoms.get_cell ()
    volume = atoms.get_volume ()
    # No information of energy, forces, stress
    iatoms, jatoms, Sij = neighbour_list(quantities='ijS',
                                        atoms=atoms,
                                        cutoff=cutoff)
    
    species = jnp.array([uniq_element[iz] for iz in atoms.numbers])
    num_nodes = crds.shape[0]
    num_edges = iatoms.shape[0]
    
    graph = jraph.GraphsTuple(nodes={"positions":jnp.array(crds),
                                     "species":species},
                                edges=dict(Sij=jnp.array(Sij)),
                                senders=jnp.array(iatoms),
                                receivers=jnp.array(jatoms),
                                n_node=jnp.array([num_nodes]),
                                n_edge=jnp.array([num_edges]),
                                globals=dict(cell=jnp.array([cell.array]), 
                                             volume=jnp.array([volume]))
                                )
    return graph



def checkpoint_save (fname, ckpt):
    with open(fname, 'wb') as fp:        
        pickle.dump (ckpt, fp)

def checkpoint_load (fname):
    with open (fname, 'rb') as fp:
        return pickle.load (fp)
    	
def check_elapsed_time(start_time:str, end_time:str):
    start_time = datetime.strptime(start_time, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.strptime(end_time, "%m/%d/%Y %H:%M:%S")
    time_difference = end_time - start_time
    return time_difference

def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.now().strftime(fmt)

def find_input_json():
    current_dir = os.getcwd()
    input_json_path = os.path.join(current_dir, 'input.json')
    if os.path.exists(input_json_path):
        return input_json_path
    else:
        return None

def mae(value1, value2):
    # value1, value2 are constant
    return jnp.abs (value1-value2)
    

def mse(value1, value2):
    # value1, value2 are constant
    return (value1-value2)**2
    
    
