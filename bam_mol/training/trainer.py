import json
import jax 
import jax.numpy as jnp 
import optax 
from bam_mol.training.train_state import TrainState 
from bam_mol.util import (import_base_model,
                          get_trajectory, 
                          checkpoint_load, 
                          checkpoint_save, 
                          date, 
                          find_input_json)

import jraph
import pickle
import e3nn_jax as e3nn
from functools import partial

# using float32
jax.config.update ("jax_enable_x64", False)


def l2_regularization (params):
    wgt, _ = jax.flatten_util.ravel_pytree (params) # or state.params instead of params
    return jnp.einsum('i,i->i', wgt, wgt).mean()


def get_edge_relative_vectors (R, data_graph):
    # iatoms ==> senders
    # jatoms ==> receivers
    iatoms = data_graph.senders
    jatoms = data_graph.receivers
    
    Rij = (R[jatoms] - R[iatoms] )

    return Rij # (num_edges, 3)


def energy_forces_fn (graphset, params, apply_fn):

    def local_energy_fn (R, params):
        Rij = get_edge_relative_vectors (R, graphset)
        energy = apply_fn({'params':params}, Rij, graphset)
        mask = jraph.get_graph_padding_mask (graphset)
        energy = energy*mask
        return energy.sum(), energy
    
    R = graphset.nodes['positions']
    
    grad, graph_energy = \
        jax.grad (local_energy_fn, argnums=(0), has_aux=True) (R, params)

    return graph_energy.reshape(-1)[:-1], \
            -grad.reshape(-1)[:-3]


def energy_fn (graphset, params, apply_fn):

    R = graphset.nodes['positions']
    Rij = get_edge_relative_vectors (R, graphset)
    graph_energy = apply_fn({'params':params}, Rij, graphset)
    mask = jraph.get_graph_padding_mask (graphset)
    graph_energy = energy*mask

    return graph_energy.reshape(-1)[:-1]


def loss_value (prd, tgt):
    """
    enr_var is 1 (constant, not trainable value) if output_irreps = 1x0e
    enr_var becomes a trainable parameter if output_irreps = 2x0e
    """
    
    # maximum atom number: 35
    #diff_enr = 10*(tgt_enr - prd_enr)/n_node
    diff = (tgt - prd)
    
    loss2 = jnp.einsum('i,i->i', diff, diff).mean()
    mae = jnp.abs (diff).mean()

    return loss2, mae


#@partial(jax.jit, static_argnames=['lambd'])
@jax.jit 
def train_enr_step (state, graphset, lambd):
    
    def loss_fn (params, apply_fn):
        predicts = energy_fn (graphset, params, apply_fn)
        targets = graphset.globals['energy'][:-1]

        loss_enr2, _ = loss_value(predicts, targets)
        loss_l2 = l2_regularization (params)
        return loss_enr2 + lambd * loss_l2
        
    grads = jax.grad (loss_fn) (state.params, state.apply_fn) 
    
    return state.apply_gradients(grads=grads) 


@jax.jit 
def train_step (state, graphset, flambda, lambd):
    
    def loss_fn (params, apply_fn):
        prd_enr, prd_frc = energy_forces_fn (graphset, params, apply_fn)
        tgt_enr = graphset.globals['energy'][:-1]
        tgt_frc = graphset.globals['forces'].reshape(-1)[:-3]
        
        loss_enr2, _ = loss_value(prd_enr, tgt_enr)
        loss_frc2, _ = loss_value (prd_frc, tgt_frc)
        loss_l2 = l2_regularization (params)
        return loss_enr2 + flambda*loss_frc2 + lambd * loss_l2
        
    grads = jax.grad (loss_fn) (state.params, state.apply_fn) 
    
    return state.apply_gradients(grads=grads) 


@jax.jit 
def get_loss (state, graphset):
        
    prd_enr, prd_frc = energy_forces_fn (graphset, state.params, state.apply_fn)
    tgt_enr, tgt_frc = graphset.globals['energy'][:-1], graphset.globals['forces'].reshape(-1)[:-3]
    loss_enr2, mae_enr = loss_value(prd_enr, tgt_enr)
    loss_frc2, mae_frc = loss_value(prd_frc, tgt_frc)
    
    return loss_enr2, mae_enr, loss_frc2, mae_frc

@jax.jit 
def get_enr_loss (state, graphset):
        
    prd_enr = energy_fn (graphset, state.params, state.apply_fn)
    tgt_enr = graphset.globals['energy'][:-1]
    loss_enr2, mae = loss_value(prd_enr, tgt_enr)
    
    return loss_enr2, mae

    
def get_dataset_loss (state, dataset, l_forces):
    if l_forces:
        loss_list = [get_loss (state, x) for x in dataset]
    else:
        loss_list = [get_enr_loss (state, x) for x in dataset]

    loss_list = jnp.array(loss_list)
    loss_enr2 = loss_list[:,0].mean()
    mae_enr = loss_list[:,1].mean()
    loss_frc2 = None
    mae_frc = None
    if l_forces:
        loss_frc2 = loss_list[:,2].mean()
        mae_frc = loss_list[:,3].mean()

    return loss_enr2, mae_enr, loss_frc2, mae_frc


def trainer (json_data):
    
    rng_seed = json_data['NN']['rng_seed']
    rng = jax.random.PRNGKey(rng_seed)

    GraphNN = import_base_model(json_data['model'])
    
    l_forces = json_data['l_forces']
    
    if l_forces:
        my_train_step = train_step 
    else:
        my_train_step = train_enr_step
    
    rng, key = jax.random.split (rng)
    fout = open(json_data['train']['fname_log'], 'w', 1)
    x_train, x_valid, uniq_element, enr_avg_per_element = \
          get_trajectory (json_data['fname_traj'],
                            json_data['ntrain'],
                            json_data['nvalid'],
                            json_data['cutoff'],
                            json_data['nbatch'],
                            json_data['element'],
                            key)
    data_info = {
        'uniq_element': uniq_element,
        'enr_avg_per_element': enr_avg_per_element
    }

    with open(json_data['train']['train_set_out'], 'wb') as f:
        pickle.dump(x_train, f)
    with open(json_data['train']['valid_set_out'], 'wb') as f:
        pickle.dump(x_valid, f)
    with open(json_data['train']['enr_ave_std_out'], 'wb') as f:
        pickle.dump(data_info, f)

    
    
    output_irreps = e3nn.Irreps ("1x0e")

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
    graphset = x_train[0]
    R = graphset.nodes['positions']
    Rij = get_edge_relative_vectors (R, graphset)
    params = model.init (key, Rij, graphset)['params']

    lr = json_data['NN']['learning_rate']
    opt_method = optax.chain (optax.clip_by_global_norm (0.5),
                              optax.clip (0.5),
                              optax.inject_hyperparams(optax.amsgrad) (learning_rate=lr))
    
    state = TrainState.create (
        apply_fn = model.apply,
        params = params,
        tx=opt_method,
        ema_decay=json_data['NN']['ema_decay']
    )
    flambda = json_data['NN']['frc_lambda']
    lambd = json_data['NN']['l2_lambda']
    print(date())
    
    decay_factor = json_data['NN']['decay_factor']
    lr_min = 0.2*lr 
    itolerate = 0
    
    loss_dict = {'train': [], 'valid': []} 
    l_ckpt_saved = False
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state,
        'uniq_element': uniq_element,
        'enr_avg_per_element': enr_avg_per_element,
        'loss': loss_dict
    }
    if json_data['NN']['restart']:
        ckpt = checkpoint_load (json_data['NN']['fname_pkl'])
        state = state.replace (params=ckpt['params'],
                               opt_state=ckpt['opt_state'])
        l_ckpt_saved = True
    
    loss_l2 = l2_regularization(state.params)
    loss_enr2_valid, mae_valid, loss_frc2_valid, _ = \
        get_dataset_loss (state, x_valid, l_forces)
    
    loss_valid_min = loss_enr2_valid + lambd * loss_l2
    if l_forces:
        loss_valid_min += flambda*loss_frc2_valid
    loss_valid_min_local = loss_valid_min
    
    if l_forces:
        print (f'                    \tTRAIN_loss_____________________________________________________________| VALID_loss', file=fout)
        line = f"MM/DD/YYYY HH/MM/SS\t {'EPOCH':7}{'LOSS':11}{'MSE_E':11}{'MAE_E':11}{'MSE_F':11}{'MAE_F':11}{'L2':10}| "
        line = line + f"{'LOSS':11}{'RMSE_E':11}{'MAE_E':11}{'RMSE_F':11}{'MAE_F':11}{'LR   ':8}"
        print (line, file=fout)
        print ('----------------------------------------------------------------------------------', file=fout)
    else:
        print (f'                    \t TRAIN_loss________________________________________| VALID_loss__________', file=fout)
        line = f"MM/DD/YYYY HH/MM/SS\t {'EPOCH':7}{'LOSS':11}{'MSE_E':11}{'MAE_E':11}{'L2':10}| "
        line = line + f"{'LOSS':11}{'RMSE_E':11}{'MAE_E':11}"
        print (line, file=fout)
        print ('------------------------------------------------------------------------------------------------', file=fout)

    #tag = 'F'
    print(date())
    nepoch = json_data['NN']['nepoch']
    for epch in range (nepoch): 
        for graphset in x_train:
            state = my_train_step (state, graphset, flambda=flambda, lambd=lambd)
       
        
        if (epch+1)%10 == 0:
            # Estimate LOSS
            loss_l2 = l2_regularization(state.params)
            loss_enr2, mae, loss_frc2, mae_frc = \
                get_dataset_loss (state, x_train, l_forces)
            loss_enr2_valid, mae_valid, loss_frc2_valid, mae_frc_valid = \
                  get_dataset_loss (state, x_valid, l_forces)
            
            loss = loss_enr2 +  lambd * loss_l2
            loss_valid = loss_enr2_valid + lambd * loss_l2
            if l_forces:
                loss += flambda*loss_frc2
                loss_valid += flambda*loss_frc2_valid

            
            loss_dict['train'].append (loss)
            loss_dict['valid'].append (loss_valid)
            rmse_E_valid = jnp.sqrt (loss_enr2_valid)
            if l_forces:
                rmse_F_valid = jnp.sqrt (loss_frc2_valid)
                line = f'{date()}\t {epch+1:<7}{loss:<11.6f}{loss_enr2:<11.6f}{mae:<11.6f}{loss_frc2:<11.6f}{mae_frc:<11.6f}{loss_l2:<11.6f}'
                line += f'{loss_valid:<11.6f}{rmse_E_valid:<11.7f}{mae_valid:<11.6f}{rmse_F_valid:<11.7f}{mae_frc_valid:<11.6f}'
            else:
                line = f'{date()}\t {epch+1:<7}{loss:<11.6f}{loss_enr2:<11.6f}{mae:<11.6f}{loss_l2:<11.6f} {loss_valid:<11.6f}{rmse_E_valid:<11.6f}{mae_valid:<11.6f}'
                

            if loss_valid < loss_valid_min:
                loss_valid_min = loss_valid
                ckpt['params'] = state.params
                ckpt['opt_state'] = state.opt_state
                ckpt['loss'] = loss_dict
                l_ckpt_saved = False
            
            if loss_valid < loss_valid_min_local:
                itolerate = 0
                loss_valid_min_local = loss_valid
            else:
                itolerate = itolerate + 1
            
            lr = optax.tree_utils.tree_get (state.opt_state, "learning_rate")
            if itolerate >= 50:
                lr_new = jnp.where (lr*decay_factor > lr_min, lr*decay_factor, lr_min)
                opt_state = optax.tree_utils.tree_set (state.opt_state, learning_rate=lr_new)
                state = state.replace (opt_state=opt_state)
                itolerate = 0
                lr = lr_new 
                loss_valid_min_local = loss_valid + 0.01
            
            line += f'{lr:<7.4f}'
            print (line, file=fout)
            
        if (epch+1)%json_data['NN']['nsave'] == 0 and \
            not l_ckpt_saved:
            checkpoint_save (json_data['NN']['fname_pkl'], ckpt)
            l_ckpt_saved = True

        if (epch+1)%20 == 0:
            train_step._clear_cache()
    

if __name__ == '__main__':
    print(date())

    input_json_path = find_input_json()

    with open(input_json_path) as f:
        json_data = json.load(f)
        trainer(json_data)
        
    print(date())
