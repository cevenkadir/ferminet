import imp
import jax.numpy as jnp
import jax
from typing import Sequence, Tuple

from .atom import Atom
from .params import NetParams


class FermiNet:
    def __init__(
        self,
        atoms: Sequence[Atom],
        n_electrons: Sequence[int],
        L: int,
        n_k: int,
        n_1: Sequence[int],
        n_2: Sequence[int],
    ):
        """
        FermiNet class.

        Args:
            atoms (`Sequence[Atom]`): The atoms to be used in the calculation.
            n_electrons (`Sequence[int]`): The number of alpha and beta electrons.
            L (`int`): The number of steps to be performed.
            n_k (`int`): The number of many-electron determinants.
            n_1 (`Sequence[int]`): The numbers of hidden units for the one-electron stream.
            n_2 (`Sequence[int]`): The numbers of hidden units for the two-electron stream.
        """

        self._atoms = atoms
        self._n_electrons = n_electrons
        self._L = L
        self._n_k = n_k
        self._n_1 = n_1
        self._n_2 = n_2

    @property
    def atoms(self) -> Sequence[Atom]:
        """`Sequence[Atom]`: The atoms in the system."""
        return self._atoms

    @property
    def n_electrons(self) -> Sequence[int]:
        """`Sequence[int]`: The number of up and down electrons."""
        return self._n_electrons

    @property
    def L(self) -> int:
        """`int`: The number of layers."""
        return self._L

    @property
    def n_k(self) -> int:
        """`int`: The number of many-electron determinants."""
        return self._n_k

    @property
    def n_1(self) -> Sequence[int]:
        """`Sequence[int]`: The number of hidden units for the one-electron stream for each layer."""
        return self._n_1

    @property
    def n_2(self) -> Sequence[int]:
        """`Sequence[int]`: The number of hidden units for the two-electron stream for each layer."""
        return self._n_2

    @property
    def n_atoms(self) -> int:
        """`int`: The number of atoms in the system."""
        return len(self.atoms)

    @property
    def n_total_electrons(self) -> int:
        """`int`: The total number of electrons in the system."""
        return sum(self.n_electrons)

    def init_params(self, seed=jax.random.PRNGKey(0)) -> NetParams:
        """`NetParams`: Initialize the network parameters."""
        return NetParams.init(
            self.n_1, self.n_2, self.n_k, self.n_electrons, len(self.atoms), seed
        )

    def init_walker_config(
        self,
        std: float,
        seed=jax.random.PRNGKey(0),
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
        """_summary_

        Args:
            atoms (`Sequence[Atom]`): List of atoms in the molecule.
            n_electrons (`Sequence[int]`): Numbers of alpha and beta electrons.
            std (`float`): Standard deviation from the atom positions.
            seed : Defaults to `jax.random.PRNGKey(0)`.

        Raises:
            ValueError: if the number of electrons does not match the total charge of the molecule.

        Returns:
            `Tuple[jnp.DeviceArray, jnp.DeviceArray]`: Positions of alpha and beta electrons in a batch with the given size.
        """

        n_electrons = jnp.asarray(self.n_electrons)  # converting to jnp.DeviceArray
        total_n_electrons = sum(
            n_electrons
        )  # total number of electrons in the molecule

        atom_positions = jnp.array(
            [atom.pos for atom in self.atoms]
        )  # positions of all atoms
        atom_charges = jnp.array(
            [atom.charge for atom in self.atoms]
        )  # charge of each atom

        total_charge = sum(atom_charges)  # total charge of the molecule

        # total atomic number of the molecule
        total_atomic_number = sum([atom.element.atomic_number for atom in self.atoms])

        # initialize spin configuration
        if total_charge == 0 and total_atomic_number == total_n_electrons:
            spin_configs = jnp.array([atom.calc_n_electrons() for atom in self.atoms])
        elif total_atomic_number - total_charge != total_n_electrons:
            raise ValueError("Number of electrons does not match total charge.")
        else:
            new_atomic_ns = jnp.array(
                [
                    atom.element.atomic_number - atom_charges[i]
                    for i, atom in enumerate(self.atoms)
                ]
            )
            spin_configs = jnp.array(
                [Atom._calc_n_electrons(sym) for sym in new_atomic_ns]
            )

        # correct the total spin projection of the molecule
        while jnp.all(jnp.sum(spin_configs, axis=0) != n_electrons):
            i = jax.random.randint(seed, [], 0, len(self.atoms))
            newseed, seed = jax.random.split(seed)

            n_alpha_i, n_beta_i = spin_configs.at[i].get()
            spin_configs = spin_configs.at[i].set([n_beta_i, n_alpha_i])

        # initialize the electron positions around the position of the nucleus of their respective atom
        alpha_electron_positions = jnp.vstack(
            jnp.repeat(jnp.array([atom_i_position]), spin_configs[i, 0], axis=0)
            for i, atom_i_position in enumerate(atom_positions)
        )
        beta_electron_positions = jnp.vstack(
            jnp.repeat(jnp.array([atom_i_position]), spin_configs[i, 1], axis=0)
            for i, atom_i_position in enumerate(atom_positions)
        )

        # add gaussian noise
        alpha_electron_positions = alpha_electron_positions + std * jax.random.normal(
            seed, shape=(n_electrons[0], 3), dtype=float
        )
        newseed, seed = jax.random.split(seed)
        beta_electron_positions = beta_electron_positions + std * jax.random.normal(
            seed, shape=(n_electrons[1], 3), dtype=float
        )

        # return the initialized walker configurations
        return alpha_electron_positions, beta_electron_positions

    def apply(
        self, params: NetParams, walker_cfg: jnp.DeviceArray, return_phi=False
    ) -> jnp.DeviceArray:
        """Calculate the natural logarithm of the modulus of the wave function for given network parameters and walker configuration.

        Args:
            params (`NetParams`): The network parameters.
            walker_cfg (`jax.numpy.DeviceArray`): The walker configuration.

        Returns:
            `jax.numpy.DeviceArray`: The natural logarithm of the modulus of the variational wave function.
        """

        # create a list that contains spin states for each electron
        spin_cfg = jnp.array(
            (True,) * self.n_electrons[0] + (False,) * self.n_electrons[1]
        )

        # get the nuclear positions of the atoms
        nuclear_positions = jnp.array([atom.pos for atom in self.atoms])

        # get a dense multi-dimensional meshgrid for the one-electron features
        pos_mgrid_for_ne = jnp.mgrid[: self.n_total_electrons, : self.n_atoms]

        # calculate h_alpha without norms
        h_alpha = walker_cfg[pos_mgrid_for_ne[0], :] - nuclear_positions
        initial_h_alpha = h_alpha.copy()

        # calculate the norms of h_alpha
        h_alpha_norms = jnp.linalg.norm(h_alpha, axis=2, keepdims=True)

        # concatanate h_alpha and h_alpha_norms
        h_alpha = jnp.concatenate((h_alpha, h_alpha_norms), axis=2)

        # resize h_alpha to (n_total_electrons, 4 * n_atoms)
        h_alpha = jnp.resize(h_alpha, (self.n_total_electrons, 4 * self.n_atoms))

        # define the spin states of the electrons in h_alpha
        h_alpha_spins = spin_cfg

        # get a dense multi-dimensional meshgrid for the two-electron features
        pos_mgrid_for_ee = jnp.mgrid[: self.n_total_electrons, : self.n_total_electrons]

        # calculate h_ee without norms
        h_alpha_beta = walker_cfg[pos_mgrid_for_ee[0], :] - walker_cfg

        h_alpha_beta_diag_indices = jnp.array(
            [jnp.ones((2,), dtype=int) * i for i in jnp.arange(self.n_total_electrons)]
        )

        h_alpha_beta = h_alpha_beta.at[
            h_alpha_beta_diag_indices[:, 0], h_alpha_beta_diag_indices[:, 1]
        ].set(100)

        # calculate the norms of h_alpha_beta
        h_alpha_beta_norms = jnp.linalg.norm(h_alpha_beta, axis=-1, keepdims=True)

        h_alpha_beta = h_alpha_beta.at[
            h_alpha_beta_diag_indices[:, 0], h_alpha_beta_diag_indices[:, 1]
        ].set(0)
        h_alpha_beta_norms = h_alpha_beta_norms.at[
            h_alpha_beta_diag_indices[:, 0], h_alpha_beta_diag_indices[:, 1]
        ].set(0)

        # concatanate h_alpha_beta and h_alpha_beta_norms
        h_alpha_beta = jnp.concatenate((h_alpha_beta, h_alpha_beta_norms), axis=2)

        # define the spin states of the electrons in h_alpha
        h_alpha_beta_spins = (
            spin_cfg[pos_mgrid_for_ee[0]],
            spin_cfg[pos_mgrid_for_ee[1]],
        )

        h_alpha, h_alpha_beta = self.exec_inter_layers(params, h_alpha, h_alpha_beta)

        lnpsi_wfs, phi_wfs = self.exec_deter_layer(params, h_alpha, initial_h_alpha)

        if return_phi:
            return lnpsi_wfs, phi_wfs
        else:
            return lnpsi_wfs

    def layer_func(
        self,
        params: NetParams,
        layer_index: int,
        h_alpha: jnp.DeviceArray,
        h_alpha_beta: jnp.DeviceArray,
    ) -> Sequence[jnp.DeviceArray]:
        """Calculate the next one- and two-electron features for a given layer.

        Args:
            params (`NetParams`): The network parameters.
            layer_index (`int`): The index of the layer.
            h_alpha (`jax.numpy.DeviceArray`): The one-electron features.
            h_alpha_beta (`jax.numpy.DeviceArray`): The two-electron features.

        Returns:
            `Sequence[jax.numpy.DeviceArray]`: The next one- and two-electron features.
        """

        # get the weights and biases for the layer
        V = params.V[layer_index]
        b = params.b[layer_index]
        W = params.W[layer_index]
        c = params.c[layer_index]

        # calculate h_alpha
        h_up = h_alpha[: self.n_electrons[0]]
        h_down = h_alpha[self.n_electrons[0] :]

        # calculate h_alpha_beta
        h_alpha_up = h_alpha_beta[:, : self.n_electrons[0]]
        h_alpha_down = h_alpha_beta[:, self.n_electrons[0] :]

        # calculate g_alpha
        g_up = jnp.mean(h_up, axis=0)
        g_down = jnp.mean(h_down, axis=0)

        # calculate g_alpha_beta
        g_alpha_up = jnp.mean(h_alpha_up, axis=1)
        g_alpha_down = jnp.mean(h_alpha_down, axis=1)

        # calculate f_alpha by concatanating h_alpha, g_alpha and g_alpha_beta
        f_alpha = jnp.concatenate(
            (
                h_alpha,
                jnp.repeat(
                    jnp.expand_dims(g_up, axis=0), self.n_total_electrons, axis=0
                ),
                jnp.repeat(
                    jnp.expand_dims(g_down, axis=0), self.n_total_electrons, axis=0
                ),
                g_alpha_up,
                g_alpha_down,
            ),
            axis=1,
        )

        # vmap the matmul function and the result itself
        matmul_vmap = jax.vmap(jnp.matmul, in_axes=(None, 0), out_axes=0)
        matmul_vmap_vmap = jax.vmap(matmul_vmap, (None, 0), out_axes=0)

        next_h_alpha = jnp.tanh(matmul_vmap(V, f_alpha) + b)
        next_h_alpha_beta = jnp.tanh(matmul_vmap_vmap(W, h_alpha_beta) + c)

        # if the layer is not the first layer
        if layer_index != 0:
            # calculate next_h_alpha by summing h_alpha and next_h_alpha
            next_h_alpha = next_h_alpha + h_alpha

            # calculate next_h_alpha_beta by summing h_alpha_beta and next_h_alpha_beta
            next_h_alpha_beta = next_h_alpha_beta + h_alpha_beta

        return next_h_alpha, next_h_alpha_beta

    def exec_inter_layers(
        self, params: NetParams, h_alpha: jnp.DeviceArray, h_alpha_beta: jnp.DeviceArray
    ) -> Sequence[jnp.DeviceArray]:
        """Execute the inter-layer calculations.

        Args:
            params (`NetParams`): The network parameters.
            h_alpha (`jax.numpy.DeviceArray`): The one-electron features.
            h_alpha_beta (`jax.numpy.DeviceArray`): The two-electron features.

            Returns:
                `Sequence[jax.numpy.DeviceArray]`: The last one- and two-electron features.
        """

        # for each layer
        for ell in range(self.L):
            # calculate the next one- and two-electron features
            h_alpha, h_alpha_beta = self.layer_func(params, ell, h_alpha, h_alpha_beta)

        return h_alpha, h_alpha_beta

    def exec_deter_layer(
        self,
        params: NetParams,
        h_alpha: jnp.DeviceArray,
        initial_h_alpha: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        """
        Execute the deterministic layer calculations.

        Args:
            params (`NetParams`): The network parameters.
            h_alpha (`jax.numpy.DeviceArray`): The one-electron features.
            initial_h_alpha (`jax.numpy.DeviceArray`): The initial one-electron features.

        Returns:
            `jax.numpy.DeviceArray`: The natural logarithm of the modulus of the wave function.
        """
        left_factor_func = (
            lambda h_alpha, w_alpha, g_alpha: jnp.dot(w_alpha, h_alpha.T) + g_alpha
        )
        left_factor_func_vmap = jax.vmap(left_factor_func, (None, 0, 0), 0)
        left_factor = left_factor_func_vmap(h_alpha, params.w_alpha, params.g_alpha)

        right_factor_func = lambda initial_h_alpha, Sigma_alpha: jnp.exp(
            -jnp.linalg.norm(
                jnp.matmul(jnp.swapaxes(initial_h_alpha, 0, 1), Sigma_alpha),
                axis=-1,
            )
        )

        right_factor_func_vmap = jax.vmap(right_factor_func, (None, 0), 0)

        right_factor = jnp.sum(
            jnp.expand_dims(params.pi_alpha, 2)
            * jnp.swapaxes(
                right_factor_func_vmap(initial_h_alpha, params.Sigma_alpha), -1, -2
            ),
            axis=-1,
        )

        # calculate one-electron orbital
        phi_wf = left_factor * right_factor

        # vmap the determinant function for each determinants
        slogdet_down_func = lambda phi_wf: jnp.linalg.slogdet(
            phi_wf[self.n_electrons[0] :, self.n_electrons[0] :]
        )
        slogdet_up_func = lambda phi_wf: jnp.linalg.slogdet(
            phi_wf[: self.n_electrons[0], : self.n_electrons[0]]
        )

        slogdet_up_func_vmap = jax.vmap(slogdet_up_func, 0, (0, 0))
        slogdet_down_func_vmap = jax.vmap(slogdet_down_func, 0, (0, 0))

        up_sign, lndet_up = slogdet_up_func_vmap(phi_wf)

        down_sign, lndet_down = slogdet_down_func_vmap(phi_wf)

        largest_lndet = jnp.amax(jnp.stack((lndet_up, lndet_down)))

        # calculate the natural logarithm of the modulus of the wave function by summing the logarithm of the determinants with the omega factors
        lnpsi_wf = largest_lndet + jnp.log(
            jnp.abs(
                jnp.sum(
                    params.omega
                    * up_sign
                    * down_sign
                    * jnp.exp(lndet_up + lndet_down - largest_lndet)
                )
            )
        )

        return lnpsi_wf, phi_wf
