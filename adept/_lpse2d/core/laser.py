import sys
import os

# sys.path.append(os.path.abspath("C:/Users/ngub/Documents/Projects/lasy"))
# sys.path.append(os.path.abspath("C:/Users/ngub/Documents/Projects/adept"))

from typing import Dict, Tuple
from jax import numpy as jnp, Array
from lasy.profiles.speckle_profile import SpeckleProfile


class Light:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.E0_source = cfg["units"]["derived"]["E0_source"]
        self.c = cfg["units"]["derived"]["c"]
        self.w0 = cfg["units"]["derived"]["w0"]
        self.dE0x = jnp.zeros((cfg["grid"]["nx"], cfg["grid"]["ny"]))
        self.x = cfg["grid"]["x"]
        self.y = cfg["grid"]["y"]
    
    def lasy_update(self, t, y, light_wave):
        
        """
        This function propagates the laser in the direction of the beam at time t for lasy's speckled
        laser profile

        :param t: time
        :param y: state variables
        :param light_wave: driver args
        :return: updated laser field
        """
        laser_profile = light_wave["laser_profile"] # lasy laser profile

        y_with_expanded_dims = self.y[None, :, None]
        x_with_expanded_dims = jnp.zeros_like(y_with_expanded_dims)
        t_with_expanded_dims = jnp.full_like(y_with_expanded_dims, t)
        E0y = laser_profile.evaluate(x_with_expanded_dims, y_with_expanded_dims, t_with_expanded_dims)[..., 0] # complex envelope

        wpe = self.w0 * jnp.sqrt(y["background_density"])
        delta_omega = 0
        k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + delta_omega) ** 2 - wpe**2 / self.w0**2)
        propagator = 1j * k0 * self.x[:, None]
        
        # Propagate the field in direction of the beam
        E0xy = E0y * jnp.exp(propagator)

        return jnp.stack([self.dE0x, E0xy], axis=-1)

    def laser_update(self, t: float, y: jnp.ndarray, light_wave: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function updates the laser field at time t

        :param t: time
        :param y: state variables
        :return: updated laser field
        """
        # method = "broadcast"
        # if method == "broadcast":
        wpe = self.w0 * jnp.sqrt(y["background_density"])[..., None]
        k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + light_wave["delta_omega"][None, None]) ** 2 - wpe**2 / self.w0**2)

        E0_static = (
            (1 + 0j - wpe**2.0 / (self.w0 * (1 + light_wave["delta_omega"][None, None])) ** 2) ** -0.25
            * self.E0_source
            * jnp.sqrt(light_wave["intensities"][None, None])
            * jnp.exp(1j * k0 * self.x[:, None, None] + 1j * light_wave["phases"][None, None])
        )
        dE0y = E0_static * jnp.exp(-1j * light_wave["delta_omega"][None, None] * self.w0 * t)
        dE0y = jnp.sum(dE0y, axis=-1)

        # elif method == "map":
        #     wpe = self.w0 * jnp.sqrt(y["background_density"])

        #     def _fn_(light_wave_item):
        #         delta_omega, intensity, initial_phase = light_wave_item
        #         k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + delta_omega) ** 2 - wpe**2 / self.w0**2)
        #         coeff = (1 + 0j - wpe**2.0 / (self.w0 * (1 + delta_omega)) ** 2) ** -0.25
        #         E0_static = (
        #             coeff
        #             * self.E0_source
        #             * jnp.sqrt(intensity)
        #             * jnp.exp(1j * k0 * self.x[:, None] + 1j * initial_phase)
        #         )
        #         dE0y = E0_static * jnp.exp(-1j * delta_omega * self.w0 * t)
        #         return dE0y

        #     dE0y = jmp(
        #         _fn_,
        #         [light_wave["delta_omega"], light_wave["intensities"], light_wave["initial_phase"]],
        #         batch_size=256,
        #     )
        #     dE0y = jnp.sum(dE0y, axis=0)

        # else:
        #     raise NotImplementedError

        return jnp.stack([self.dE0x, dE0y], axis=-1)

    def calc_ey_at_one_point(self, t: float, density: Array, light_wave: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function is used to calculate the coherence time of the laser

        :param t: time
        :param y: state variables
        :return: updated laser field
        """

        wpe = self.w0 * jnp.sqrt(density)[None, 0, 0]
        k0 = self.w0 / self.c * jnp.sqrt((1 + 0j + light_wave["delta_omega"]) ** 2 - wpe**2 / self.w0**2)
        E0_static = (
            (1 + 0j - wpe**2.0 / (self.w0 * (1 + light_wave["delta_omega"])) ** 2) ** -0.25
            * self.E0_source
            * jnp.sqrt(light_wave["intensities"])
            * jnp.exp(1j * k0 * self.x[0] + 1j * light_wave["phases"])
        )
        dE0y = E0_static * jnp.exp(-1j * light_wave["delta_omega"] * self.w0 * t)
        return jnp.sum(dE0y, axis=0)
