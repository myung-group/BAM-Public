from typing import Any
from collections.abc import Callable

import optax

import jax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT

class TrainState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    ema_decay: float 

    def apply_gradients (self, *, grads, **kwargs):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt
        )
        
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        new_params_with_opt = jax.tree_util.tree_map (
            lambda x, y: x*self.ema_decay + y*(1-self.ema_decay), params_with_opt, new_params_with_opt
        )

        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        
        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
            )
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
        
        return cls(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
            )
