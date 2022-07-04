import jax.numpy as jnp
import jax
from typing import Sequence

from .params import NetParams
from .network import FermiNet


class Hamiltonian:
    def __init__(self, ferminet: FermiNet):
        """Hamiltonian class for the FermiNet.

        Args:
            ferminet: FermiNet object.
        """

        self._ferminet = ferminet

    @property
    def ferminet(self) -> FermiNet:
        return self._ferminet

    def calc_grads(
        self, params: NetParams, walker_cfg: jnp.DeviceArray
    ) -> Sequence[jnp.DeviceArray]:
        """Calculate gradients of the natural logarithm of the absolute value of the wave function with respect to the given walker configuration.

        Args:
            params (`NetParams`): The parameters of the network.
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.

        Returns:
            `Sequence[jax.numpy.DeviceArray]`: A list of gradients of the natural logarithm of the absolute value of the wave function with respect to the given walker configuration.
        """

        # prepare the first and second order Jacobians

        log_psi_jac = jax.grad(self.ferminet.apply, argnums=1)
        log_psi_jac_jac = jax.jacfwd(log_psi_jac, argnums=1)

        first_jac = log_psi_jac(params, walker_cfg)
        second_jac = log_psi_jac_jac(params, walker_cfg)

        # print(second_jac)

        second_jac = jnp.vstack(
            [
                second_jac[i, jnp.arange(3), i, jnp.arange(3)]
                for i in jnp.arange(self.ferminet.n_total_electrons)
            ]
        )

        return first_jac, second_jac

    def calc_kinetic_energy(
        self, params: NetParams, walker_cfg: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Calculate the kinetic part of the local energy.

        Args:
            params (`NetParams`): The parameters of the network.
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.

        Returns:
            `jax.numpy.DeviceArray`: The kinetic part of the local energy.
        """

        first_grad, second_grad = self.calc_grads(params, walker_cfg)

        kinetic_energy = -0.5 * (jnp.sum(first_grad**2) + jnp.sum(second_grad))

        return kinetic_energy

    def calc_potential_energy(
        self, walker_cfg: jnp.DeviceArray
    ) -> Sequence[jnp.DeviceArray]:
        """Calculate the potential part of the local energy.

        Args:
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.

        Returns:
            `Sequence[jax.numpy.DeviceArray]`: The potential part of the local energy.
        """

        # get a dense multi-dimensional meshgrid for the two-electron features
        pos_mgrid_for_ee = jnp.mgrid[
            : self.ferminet.n_total_electrons, : self.ferminet.n_total_electrons
        ]

        ee_vectors = walker_cfg[pos_mgrid_for_ee[0], :] - walker_cfg
        first_term = 1 / jnp.linalg.norm(ee_vectors, axis=2)
        first_term = jnp.sum(first_term[jnp.triu_indices_from(first_term, k=1)])

        # get the nuclear positions of the atoms
        nuclear_positions = jnp.array([atom.pos for atom in self.ferminet.atoms])
        atomic_numbers = jnp.array(
            [atom.element.atomic_number for atom in self.ferminet.atoms]
        )

        # get a dense multi-dimensional meshgrid for the one-electron features
        pos_mgrid_for_ne = jnp.mgrid[
            : self.ferminet.n_total_electrons, : self.ferminet.n_atoms
        ]

        ne_vectors = walker_cfg[pos_mgrid_for_ne[0], :] - nuclear_positions

        second_term = -atomic_numbers[pos_mgrid_for_ne[1]] / jnp.linalg.norm(
            ne_vectors, axis=2
        )
        second_term = jnp.sum(second_term)

        # get a dense multi-dimensional meshgrid for the two-nuclei features
        pos_mgrid_for_nn = jnp.mgrid[: self.ferminet.n_atoms, : self.ferminet.n_atoms]
        nn_vectors = nuclear_positions[pos_mgrid_for_nn[0], :] - nuclear_positions

        third_term = (
            atomic_numbers[pos_mgrid_for_nn[0]]
            * atomic_numbers[pos_mgrid_for_nn[1]]
            / jnp.linalg.norm(nn_vectors, axis=2)
        )
        third_term = jnp.sum(third_term[jnp.triu_indices_from(third_term, k=1)])

        return first_term, second_term, third_term

    def calc_local_energy(
        self, params: NetParams, walker_cfg: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Calculate the local energy.

        Args:
            params (`NetParams`): The parameters of the network.
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.

        Returns:
            `jax.numpy.DeviceArray`: The local energy.
        """

        kinetic_energy = self.calc_kinetic_energy(params, walker_cfg)
        total_potential_energy = sum(self.calc_potential_energy(walker_cfg))

        total_energy = kinetic_energy + total_potential_energy

        return total_energy
