import jax.numpy as jnp
import jax
import chex

from .network import FermiNet


class Sampler:
    def __init__(self, ferminet:FermiNet, n_step:int, std:float):
        self._ferminet = ferminet
        self._n_step = n_step
        self._std = std
    
    @property
    def ferminet(self)->FermiNet:
        """(`FermiNet`): The FermiNet instance."""
        return self._ferminet
    
    @property
    def n_step(self)->int:
        """(`int`): The number of steps in each sampling."""
        return self._n_step
    
    @property
    def std(self)->float:
        """(`float`): The standard deviation of the Gaussian noise in each step for each dimension."""
        return self._std

    def next_pretrain_sample(self, p, calc_wfs,n_electrons, walker_cfg:jnp.DeviceArray, seed=jax.random.PRNGKey(0)):
      for i in range(self.n_step):
          subseed, seed = jax.random.split(seed)
          candidate_walker_cfg = walker_cfg + self.std*jax.random.normal(seed, walker_cfg.shape)
          

          hf_wfs = calc_wfs(candidate_walker_cfg, n_electrons)

          new_p = jnp.prod(jnp.diag(hf_wfs**2))
          
          p_move = new_p/p
          
          alpha = jax.random.uniform(subseed)
          
          p = jnp.where(p_move>alpha, new_p, p)
          walker_cfg = jnp.where(p_move>alpha, candidate_walker_cfg, walker_cfg)
          
      return walker_cfg, p, hf_wfs

    def next_sample(self, log_psi:jnp.DeviceArray, params:chex.dataclass, walker_cfg:jnp.DeviceArray, seed=jax.random.PRNGKey(0)):
        """Generate the next sample.
        
        Args:
            log_psi (`jax.numpy.DeviceArray`): The logarithm of the modulus of the wavefunction.
            params (`chex.dataclass`): The parameters of the network.
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.
            seed (`jax.random.PRNGKey`): The seed for the random number generator.
            
        Returns:
            `jax.numpy.DeviceArray`: The next sample.
        """
        
        for i in range(self.n_step):
            subseed, seed = jax.random.split(seed)
            candidate_walker_cfg = walker_cfg + self.std*jax.random.normal(seed, walker_cfg.shape)
            
            new_log_psi=self.ferminet.apply(params,candidate_walker_cfg)
            
            p_move = jnp.exp(2*new_log_psi - 2*log_psi)
            
            alpha = jax.random.uniform(subseed)
            
            log_psi = jnp.where(p_move>alpha, new_log_psi, log_psi)
            walker_cfg = jnp.where(p_move>alpha, candidate_walker_cfg, walker_cfg)

        O = jax.grad(self.ferminet.apply, argnums=0)(params,walker_cfg)
            
        return walker_cfg, log_psi, O
        