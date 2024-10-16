import json
import pickle    
import jax
import jax.numpy as jnp     

import jraph 
import e3nn_jax as e3nn
from bam_del.training.trainer import (energy_forces_fn,
                                      get_edge_relative_vectors)
from bam_del.util import (import_base_model, 
                           get_trajectory,
                           get_trajectory_test, 
                           date, 
                           find_input_json, 
                           mae, 
                           mse)



def get_energy(data_list, 
               enr_avg_per_element, 
               params, 
               apply_fn):
    
    prd_enr = []
    prd_frc = []
    for graphset in data_list:
        #species = graphset.nodes['species']
        #node_enr_avg = jnp.array([enr_avg_per_element[int(iz)] for iz in species])
        #graph_enr_avg = e3nn.scatter_sum(node_enr_avg, nel=graphset.n_node)[:-1]

        enr, frc = \
              energy_forces_fn(graphset, params, apply_fn)
        
        prd_enr += list(jnp.array([enr.reshape(-1)]))
        prd_frc += list(jnp.array([frc.reshape(-1)]))

    prd_enr = jnp.array(prd_enr).reshape(-1)
    prd_frc = jnp.array(prd_frc).reshape(-1)

    return prd_enr,  prd_frc
    

def get_exact(data_list, model_ckpt):

    enr_avg_per_element = model_ckpt['enr_avg_per_element']
    
    exact_enr = []
    exact_frc = []
    
    for graph_data in data_list:
        #species = graph_data.nodes['species']
        #node_enr_avg = jnp.array([enr_avg_per_element[int(iz)] for iz in species])
        #graph_enr_avg = e3nn.scatter_sum(node_enr_avg, nel=graph_data.n_node)[:-1]
        
        exact_enr += list(jnp.array(graph_data.globals['energy'].reshape(-1)[:-1]))
        exact_frc += list(jnp.array(graph_data.globals['forces'].reshape(-1)[:-3]))

    return jnp.array(exact_enr).reshape(-1), \
           jnp.array(exact_frc).reshape(-1), \



def predictor (rng, x_data, model_ckpt, json_data, fout, do_print, data_type):

    
    output_irreps = e3nn.Irreps ("1x0e")
    GraphNN = import_base_model(json_data['model'])
    hidden_irreps = e3nn.Irreps (json_data['hidden_irreps'])
    model = GraphNN (json_data['cutoff'],
                     json_data['avg_num_neighbors'],
                     json_data['num_species'],
                     num_basis_func=json_data['num_radial_basis'],
                     hidden_irreps=hidden_irreps,
                     nlayers = json_data['nlayers'],
                     features_dim = json_data['features_dim'],
                     output_irreps=output_irreps)

    rng, key = jax.random.split (rng)
    graphset = x_data[0]
    R = graphset.nodes['positions']
    Rij = get_edge_relative_vectors (R, graphset)
    params = model.init (key, Rij, graphset)['params']
    
    apply_fn = jax.jit(model.apply)

    E_EXACT, F_EXACT = get_exact(x_data, model_ckpt)
    enr_avg_per_element = model_ckpt['enr_avg_per_element']
    
    Enr_mean, Frc_mean = get_energy(x_data, 
                             enr_avg_per_element, 
                             model_ckpt['params'],
                             apply_fn)
        
    ## Correction in Shifted Energy
    e_corr = Enr_mean.mean() - E_EXACT.mean()
    #Enr_mean -= e_corr

    data_to_pickle = (E_EXACT , Enr_mean)
    
    E_EXACT = jnp.sqrt(E_EXACT)
    Enr_mean = jnp.sqrt(Enr_mean)

    if data_type == 'train':
        with open(json_data['predict']['predict_train_out'], 'wb') as f:
            pickle.dump(data_to_pickle, f)
    elif data_type == 'valid':
        with open(json_data['predict']['predict_valid_out'], 'wb') as f:
            pickle.dump(data_to_pickle, f)
    else:
        with open(json_data['predict']["test"]["predict_test_out"], 'wb') as f:
            pickle.dump(data_to_pickle, f)
    
    if do_print:
        print(date())
        print(f'                   \tPREDICTED______________________| EXACT_______________', file=fout)
        print(f"MM/DD/YYYY HH/MM/SS\t {'DATA':7}{'MAE_E':11}{'E':12}| {'E':11}", file=fout) 
        print('---------------------------------------------------------------------------', file=fout)
        mae_enr = jax.vmap (mae) (E_EXACT, Enr_mean)
        #mae_frc = jax.vmap (mae) (F_EXACT, Frc_mean)
        mse_enr = jax.vmap (mse) (E_EXACT, Enr_mean)
        #mse_frc = jax.vmap (mse) (F_EXACT, Frc_mean)
        n_e = mae_enr.shape[0]

        for i in range(n_e):
            print(f'{date()}\t {i+1:<7}{mae_enr[i]:<11.6f}{Enr_mean[i]:<14.6f}{E_EXACT[i]:<12.6f}', file=fout)

        rmse_enr = jnp.sqrt(mse_enr.mean())
        rmse_frc = jnp.sqrt(mse_frc.mean())
        print(f'{date()}\t Corrected_E: {e_corr}', file=fout)
        print(f'{date()}\t  MAE_E : {mae_enr.mean()}',file=fout)
        print(f'{date()}\t RMSE_E : {rmse_enr}',file=fout)

        print(date())


if __name__ == '__main__':
    print(date())

    input_json_path = find_input_json()
    
    with open(input_json_path) as f:
        json_data = json.load(f)
        rng_seed = json_data['NN']['rng_seed']
        rng = jax.random.PRNGKey(rng_seed)
        rng, key = jax.random.split (rng)

        fname_model_pkl = json_data['NN']["fname_pkl"]
        fd_ckpt = open(fname_model_pkl, 'rb')
        model_ckpt = pickle.load(fd_ckpt)

        if json_data['predict']['start']:
            print(date(), 'predictor')
            x_train, x_valid, _, _ = \
            get_trajectory (json_data['fname_traj'],
                            json_data['ntrain'],
                            json_data['nvalid'],
                            json_data['cutoff'],
                            json_data['predict']['nbatch'],
                            json_data['element'],
                            rng = key)
            fout = open ('delta_energy_predict_valid.out', 'w', 1)
            predictor (rng, x_train, model_ckpt, json_data, fout, False, 'train')
            predictor (rng, x_valid, model_ckpt, json_data, fout, True, 'valid')
            fout.close ()

        if json_data["predict"]["test"]["start"]:
            print(date(), 'predictor:test')
            dataset = get_trajectory_test (fname = json_data["predict"]["test"]["fname_traj"],
                    ndata = json_data["predict"]["test"]["ndata"],
                    cutoff = json_data['cutoff'],
                    nbatch = json_data['predict']['nbatch'],
                    model_ckpt = model_ckpt,
                    )
            with open(json_data['predict']['test']['test_set_out'], 'wb') as f:
                pickle.dump(dataset, f)
            fout = open (json_data["predict"]["test"]["fname_log"], 'w', 1)
            predictor (rng, dataset, model_ckpt, json_data, fout, True, 'test')
            fout.close()

    print(date())
