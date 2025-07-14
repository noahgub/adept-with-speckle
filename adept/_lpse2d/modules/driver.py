from typing import Dict
import json, tempfile

from jax import numpy as jnp, Array, tree_util as jtu
from jax.random import PRNGKey

import equinox as eqx
import numpy as np
import mlflow

from adept.utils import download_from_s3

from lasy.profiles.speckle_profile import SpeckleProfile


def load(cfg: Dict, DriverModule: eqx.Module) -> eqx.Module:
    filename = cfg["drivers"]["E0"]["file"]
    with tempfile.TemporaryDirectory() as td:
        # download the file if it is on s3
        if "s3" in filename:
            from os.path import join

            print(filename)

            cfg["drivers"]["E0"]["file"] = join(td, filename.split("/")[-1])
            cfg["drivers"]["E0"]["file"] = download_from_s3(filename, cfg["drivers"]["E0"]["file"])
        else:
            cfg["drivers"]["E0"]["file"] = filename

        # load the model
        if "pkl" in cfg["drivers"]["E0"]["file"]:
            loaded_model = DriverModule(cfg)
            with open(cfg["drivers"]["E0"]["file"], "rb") as f:
                import pickle

                _loaded_pickle_ = pickle.load(f)
                loaded_model = eqx.tree_at(
                    lambda tree: tree.intensities,
                    loaded_model,
                    replace=_loaded_pickle_["E0"]["intensities"],
                )
                loaded_model = eqx.tree_at(
                    lambda tree: tree.phases,
                    loaded_model,
                    replace=_loaded_pickle_["E0"]["phases"],
                )
        elif "eqx" in cfg["drivers"]["E0"]["file"]:
            with open(cfg["drivers"]["E0"]["file"], "rb") as f:
                # read the model config
                model_cfg = json.loads(f.readline().decode())
                cfg["drivers"]["E0"]["params"] = model_cfg
                model = DriverModule(cfg)

                loaded_model = eqx.tree_deserialise_leaves(f, model)
        else:
            raise NotImplementedError(f"File type not recognized: {filename}. Must be .pkl or .eqx")

    return loaded_model


def choose_driver(shape: str) -> eqx.Module:
    if shape == "uniform":
        return UniformDriver

    elif shape == "gaussian":
        return GaussianDriver

    elif shape == "lorentzian":
        return LorentzianDriver

    elif shape == "arbitrary":
        return ArbitraryDriver

    elif shape == "speckled":
        return SpeckledDriver

    else:
        raise NotImplementedError(f"Amplitude shape -- {shape} -- not implemented")


class UniformDriver(eqx.Module):
    intensities: Array
    delta_omega: Array
    phases: Array
    envelope: Dict

    def __init__(self, cfg: Dict):
        super().__init__()

        driver_cfg = cfg["drivers"]["E0"]
        self.intensities = jnp.array(np.ones(cfg["drivers"]["E0"]["num_colors"]))
        self.delta_omega = jnp.linspace(
            -driver_cfg["delta_omega_max"],
            driver_cfg["delta_omega_max"],
            driver_cfg["num_colors"],
        )
        phase_rng = np.random.default_rng(seed=cfg["drivers"]["E0"]["params"]["phases"]["seed"])
        self.phases = jnp.array(phase_rng.uniform(-1, 1, driver_cfg["num_colors"]))
        self.envelope = driver_cfg["derived"]

    def save(self, filename: str) -> None:
        """
        Save the model to a file

        Parameters
        ----------
        filename : str
            The name of the file to save the model to

        """

        with open(filename, "wb") as f:
            model_cfg_str = json.dumps(self.model_cfg)
            f.write((model_cfg_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.intensities / jnp.sum(self.intensities)

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope

        return state, args


class SpeckledDriver(UniformDriver):
    model_cfg: Dict
    wavelength: jnp.ndarray
    energy: jnp.ndarray
    focal_length: jnp.ndarray
    beam_aperture: jnp.ndarray
    n_beamlets_x: jnp.ndarray
    n_beamlets_y: jnp.ndarray
    relative_laser_bandwidth: jnp.ndarray
    ssd_phase_modulation_amplitude: jnp.ndarray
    ssd_number_color_cycles: jnp.ndarray
    ssd_transverse_bandwidth_distribution: jnp.ndarray

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.model_cfg = cfg["drivers"]["E0"]["params"]
        driver_cfg = cfg["drivers"]["E0"]

        # Lasy parameters
        self.wavelength = jnp.array(driver_cfg["lasy_params"]["wavelength"])
        self.energy = jnp.array(driver_cfg["lasy_params"]["energy"])
        self.focal_length = jnp.array(driver_cfg["lasy_params"]["focal_length"])
        self.beam_aperture = jnp.array(driver_cfg["lasy_params"]["beam_aperture"])
        self.n_beamlets_x = jnp.array(driver_cfg["lasy_params"]["n_beamlets_x"]).item()
        self.n_beamlets_y = jnp.array(driver_cfg["lasy_params"]["n_beamlets_y"]).item()
        self.relative_laser_bandwidth = jnp.array(driver_cfg["lasy_params"]["relative_laser_bandwidth"])
        self.ssd_phase_modulation_amplitude = jnp.array(driver_cfg["lasy_params"]["ssd_phase_modulation_amplitude"])
        self.ssd_number_color_cycles = jnp.array(driver_cfg["lasy_params"]["ssd_number_color_cycles"])
        self.ssd_transverse_bandwidth_distribution = jnp.array(
            driver_cfg["lasy_params"]["ssd_transverse_bandwidth_distribution"]
        )

        # Unbound for optimization
        if self.model_cfg["relative_laser_bandwidth"]["learned"]:
            self.relative_laser_bandwidth = self.bound_param(
                self.relative_laser_bandwidth, "relative_laser_bandwidth", False
            )
        if self.model_cfg["ssd_phase_modulation_amplitude"]["learned"]:
            self.ssd_phase_modulation_amplitude = self.bound_param(
                self.ssd_phase_modulation_amplitude,
                "ssd_phase_modulation_amplitude",
                False,
            )
        if self.model_cfg["ssd_number_color_cycles"]["learned"]:
            self.ssd_number_color_cycles = self.bound_param(
                self.ssd_number_color_cycles, "ssd_number_color_cycles", False
            )
        if self.model_cfg["ssd_transverse_bandwidth_distribution"]["learned"]:
            self.ssd_transverse_bandwidth_distribution = self.bound_param(
                self.ssd_transverse_bandwidth_distribution,
                "ssd_transverse_bandwidth_distribution",
                False,
            )

    def bound_param(self, param, param_name, bound):
        """
        Bound or undbound parameter

        :param param: parameter to bound or unbound
        :param param_name: name of the parameter
        :param bounded: if true, parameter will be bounded, if false, it will be unbounded
        :return: bounded or unbounded parameter

        """
        if bound:
            return self.model_cfg[param_name]["a"] + self.model_cfg[param_name]["b"] * jnp.tanh(param)
        else:
            return jnp.arctanh((param - self.model_cfg[param_name]["a"]) / self.model_cfg[param_name]["b"])

    def get_partition_spec(self):
        """
        Get the partition spec for the model

        Phase modulation amplitude and relative bandwidth can be learned

        Returns
        -------
        filter_spec : pytree with the same structure as the model

        """
        filter_spec = jtu.tree_map(lambda _: False, self)

        # if self.model_cfg["wavelength"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.wavelength, filter_spec, replace=True)
        # if self.model_cfg["energy"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.energy, filter_spec, replace=True)
        # if self.model_cfg["focal_length"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.focal_length, filter_spec, replace=True)
        # if self.model_cfg["beam_aperture"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.beam_aperture, filter_spec, replace=True)
        # if self.model_cfg["n_beamlets_x"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.n_beamlets_x, filter_spec, replace=True)
        # if self.model_cfg["n_beamlets_y"]["learned"]:
        #     filter_spec = eqx.tree_at(lambda tree: tree.n_beamlets_y, filter_spec, replace=True)
        if self.model_cfg["relative_laser_bandwidth"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.relative_laser_bandwidth, filter_spec, replace=True)
        if self.model_cfg["ssd_phase_modulation_amplitude"]["learned"]:
            filter_spec = eqx.tree_at(
                lambda tree: tree.ssd_phase_modulation_amplitude,
                filter_spec,
                replace=True,
            )
        if self.model_cfg["ssd_number_color_cycles"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.ssd_number_color_cycles, filter_spec, replace=True)
        if self.model_cfg["ssd_transverse_bandwidth_distribution"]["learned"]:
            filter_spec = eqx.tree_at(
                lambda tree: tree.ssd_transverse_bandwidth_distribution,
                filter_spec,
                replace=True,
            )

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:

        # Bound for speckled profile instantiation
        bdd_relative_laser_bandwidth = self.relative_laser_bandwidth
        bdd_ssd_phase_modulation_amplitude = self.ssd_phase_modulation_amplitude
        bdd_ssd_number_color_cycles = self.ssd_number_color_cycles
        bdd_ssd_transverse_bandwidth_distribution = self.ssd_transverse_bandwidth_distribution

        if self.model_cfg["relative_laser_bandwidth"]["learned"]:
            bdd_relative_laser_bandwidth = self.bound_param(
                self.relative_laser_bandwidth, "relative_laser_bandwidth", True
            )
        if self.model_cfg["ssd_phase_modulation_amplitude"]["learned"]:
            bdd_ssd_phase_modulation_amplitude = self.bound_param(
                self.ssd_phase_modulation_amplitude,
                "ssd_phase_modulation_amplitude",
                True,
            )
        if self.model_cfg["ssd_number_color_cycles"]["learned"]:
            bdd_ssd_number_color_cycles = self.bound_param(
                self.ssd_number_color_cycles, "ssd_number_color_cycles", True
            )
        if self.model_cfg["ssd_transverse_bandwidth_distribution"]["learned"]:
            bdd_ssd_transverse_bandwidth_distribution = self.bound_param(
                self.ssd_transverse_bandwidth_distribution,
                "ssd_transverse_bandwidth_distribution",
                True,
            )

        laser_profile = SpeckleProfile(
            self.wavelength,
            (1, 0),  # polarization (along x)
            self.energy,
            self.focal_length,
            tuple(self.beam_aperture),
            [self.n_beamlets_x, self.n_beamlets_y],
            "FM SSD",  # temporal smoothing type
            bdd_relative_laser_bandwidth,
            tuple(bdd_ssd_phase_modulation_amplitude),
            tuple(bdd_ssd_number_color_cycles),
            tuple(bdd_ssd_transverse_bandwidth_distribution),
        )

        args["drivers"]["E0"] = {"laser_profile": laser_profile} | self.envelope

        return state, args


class ArbitraryDriver(UniformDriver):
    model_cfg: Dict

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        driver_cfg = cfg["drivers"]["E0"]
        self.model_cfg = cfg["drivers"]["E0"]["params"]
        if self.model_cfg["amplitudes"]["init"] == "random":
            self.intensities = jnp.array(np.random.uniform(-1, 1, driver_cfg["num_colors"]))
        elif self.model_cfg["amplitudes"]["init"] == "uniform":
            self.intensities = jnp.ones(driver_cfg["num_colors"])
        else:
            raise NotImplementedError(
                f"Initialization type -- {self.model_cfg['amplitudes']['init']} -- not implemented"
            )

    def scale_intensities(self, intensities):
        if self.model_cfg["amplitudes"]["activation"] == "linear":
            ints = 0.5 * (jnp.tanh(intensities) + 1.0)
        elif self.model_cfg["amplitudes"]["activation"] == "log":
            ints = 3 * (jnp.tanh(intensities) + 1.0) - 3
            ints = 10**ints
        elif self.model_cfg["amplitudes"]["activation"] == "log-3wide":
            ints = -1.5 * (jnp.tanh(intensities) + 1.0)  # from 0 to -3
            ints = 10**ints
        else:
            raise NotImplementedError(
                f"Amplitude Output type -- {self.model_cfg['amplitudes']['activation']} -- not implemented"
            )

        return ints

    def get_partition_spec(self):
        """
        Get the partition spec for the model

        Only amplitudes and phases can be learned

        Returns
        -------
        filter_spec : pytree with the same structure as the model

        """
        filter_spec = jtu.tree_map(lambda _: False, self)

        if self.model_cfg["amplitudes"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.intensities, filter_spec, replace=True)

        if self.model_cfg["phases"]["learned"]:
            filter_spec = eqx.tree_at(lambda tree: tree.phases, filter_spec, replace=True)

        return filter_spec

    def __call__(self, state: Dict, args: Dict) -> tuple:
        intensities = self.scale_intensities(self.intensities)
        intensities = intensities / jnp.sum(intensities)

        args["drivers"]["E0"] = {
            "delta_omega": self.delta_omega,
            "phases": jnp.tanh(self.phases) * jnp.pi,
            "intensities": intensities,
        } | self.envelope

        return state, args


class GaussianDriver(UniformDriver):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.intensities = jnp.array(
            2
            * np.log(2)
            / delta_omega_max
            / np.sqrt(np.pi)
            * np.exp(-4 * np.log(2) * (self.delta_omega / delta_omega_max) ** 2.0)
        )


class LorentzianDriver(UniformDriver):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        delta_omega_max = cfg["drivers"]["E0"]["delta_omega_max"]
        self.intensities = jnp.array(
            1 / np.pi * (delta_omega_max / 2) / (self.delta_omega**2.0 + (delta_omega_max / 2) ** 2.0)
        )
