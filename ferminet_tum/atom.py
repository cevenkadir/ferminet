import jax.numpy as jnp
import numpy as np
import chex
import mendeleev
import re
from typing import Sequence


@chex.dataclass
class Atom:
    """Atom class for the FermiNet.

    Args:
        id (`str`): The element identifier. For example, `C` or `6` for a Carbon atom.
        pos (`jnp.DeviceArray`): The position of the atom's nucleus.
        charge (`int`): The charge of the atom.
    """

    id: str  # element identifier
    pos: chex.ArrayDevice  # position of the atom
    charge: int = 0  # charge of the atom

    @property
    def element(self) -> mendeleev.models.Element:
        """`mendeleev.models.Element`: Mendeleev element object regarding to `id` of `Atom` object."""
        return mendeleev.element(self.id)

    def calc_e_cfg(self) -> Sequence[jnp.DeviceArray]:
        """Calculate electron configuration of `Atom` object as a sequence of arrays.

        Returns:
            `Sequence[jax.numpy.DeviceArray]`: Electron configuration of `Atom` object.

        For example, for `Atom` object with `id` `C`, the returned electron configuration is:
        >>> Atom(id='C', pos=jnp.array([0.0, 0.0, 0.0])).calc_e_cfg()
            [
                DeviceArray([[True, True]], dtype=bool),
                DeviceArray([[True, True]], dtype=bool),
                DeviceArray([
                    [True, False],
                    [True, False],
                    [False, False]],dtype=bool)
            ]
        """

        # define subshells
        ells = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

        # calculate orbital quantum numbers
        orbital_arr_shapes = {key: (2 * ells[key] + 1, 2) for key in ells.keys()}

        # get the electron configuration of Atom object in string format, using mendeleev library
        econf = self.element.econf

        # replace element symbol with the corresponding quantum numbers
        while econf.startswith("["):
            splitted_econf = econf.split()

            sym = splitted_econf[0][1:-1]

            next_econf = mendeleev.element(sym).econf

            econf = next_econf + " " + " ".join(splitted_econf[1:])

        # split the string into substrings
        splitted_econf = econf.split()

        # initialize the electron configuration array
        cfg = []

        # loop over the substrings to get the electron configuration array
        for i, orbital_i in enumerate(splitted_econf):
            first, second, n_spin = re.split(r"([A-Za-z]+)", orbital_i)
            first = int(first)
            n_spin = int(n_spin) if n_spin != "" else 1

            if i == len(splitted_econf) - 1:
                cfg_i = jnp.zeros(orbital_arr_shapes[second], dtype=bool)

                for j in range(n_spin):
                    to_fill_position_j = (j % cfg_i.shape[0], j // cfg_i.shape[0])
                    cfg_i = cfg_i.at[to_fill_position_j].set(1)
            else:
                cfg_i = jnp.ones(orbital_arr_shapes[second], dtype=bool)

            cfg.append(cfg_i)

        return cfg

    def calc_n_electrons(self) -> jnp.DeviceArray:
        """Calculate number of electrons of `Atom` object in terms of alpha and beta electrons.

        Returns:
            jax.numpy.DeviceArray: Number of alpha and beta electrons of `Atom` object.
        """
        n_electrons_per_subshell = [
            jnp.sum(subshell, axis=0) for subshell in self.calc_e_cfg()
        ]
        n_electrons_per_subshell = jnp.array(n_electrons_per_subshell)

        return jnp.sum(n_electrons_per_subshell, axis=0)
