from typing import Dict
import sys
import logging, os
from contextlib import redirect_stdout, redirect_stderr
from scipy.optimize import OptimizeResult
import argparse
from jax import config

config.update("jax_enable_x64", True)
ml_for_lpi_path = "/global/homes/n/ngub/adept"
sys.path.append(os.path.abspath(ml_for_lpi_path))
from ml_for_lpi.ml4tpd import TPDModule

# logger = logging.getLogger(__name__)

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run_one_val_and_grad(run_id: str, _cfg_path: str, epoch: int):
    """
    Runs a single val and grad step.

    This function calculates the total electrostatic energy
    in the box and the gradient of the total electrostatic energy with respect to
    the laser parameters.

    Args:
        run_id: str: The run id
        _cfg_path: str: The config path

    Returns:
        val, grad: Tuple: The value and the gradient
    """
    import yaml
    import equinox as eqx
    import mlflow
    import numpy as np

    from adept import ergoExo

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    exo = ergoExo(mlflow_run_id=run_id, mlflow_nested=True)
    modules = exo.setup(cfg, TPDModule)
    diff_modules, static_modules = {}, {}

    diff_modules["laser"], static_modules["laser"] = eqx.partition(
        modules["laser"], modules["laser"].get_partition_spec()
    )

    val, grad, (sol, ppo, _) = exo.val_and_grad(diff_modules, args={"static_modules": static_modules})

    laser_driver_object = grad["laser"]
    static_laser_driver_object = static_modules["laser"]
    model_cfg = static_laser_driver_object.model_cfg  # Get from static modules so not None
    metrics_to_log = {}
    for param_name, param_config in model_cfg.items():
        if param_config.get("learned"):
            if getattr(laser_driver_object, param_name) is None:
                break
            value_to_log = np.linalg.norm(getattr(laser_driver_object, param_name))
            metrics_to_log[f"Grad of {param_name}"] = float(value_to_log)
            # print(f"Grad norm of '{param_name}': {value_to_log}")
    mlflow.log_metrics(metrics_to_log, step=epoch)

    return val, grad


def calc_loss_and_grads(modules: Dict, epoch: int, orig_cfg: Dict):
    """
    This is a wrapper around the run_one_val_and_grad function.

    It logs the loss and the gradient norm to mlflow.

    Args:
        modules: Dict: The modules
        epoch: int: The epoch
        parent_run_id: str: The parent run id
        orig_cfg: Dict: The original config

    Returns:
        val, flat_grad, grad: Tuple: The value, the flattened gradient, and the pytree gradient


    """
    import tempfile, yaml

    import numpy as np
    import mlflow
    from jax.flatten_util import ravel_pytree

    with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as _td:
        module_path = os.path.join(_td, f"laser.eqx")
        modules["laser"].save(module_path)
        orig_cfg["drivers"]["E0"]["file"] = module_path
        orig_cfg["grid"]["dt"] = f"{np.random.uniform(1, 3):.3f}fs"
        with open(_cfg_path := os.path.join(_td, "config.yaml"), "w") as fi:
            yaml.dump(orig_cfg, fi)

        with mlflow.start_run(nested=True, run_name=f"epoch-{epoch}") as nested_run:
            pass

        # Capture the output into a file
        output_file = os.path.join(_td, f"stdout_stderr.txt")
        with open(output_file, "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                val, grad = run_one_val_and_grad(run_id=nested_run.info.run_id, _cfg_path=_cfg_path, epoch=epoch)

        mlflow.log_artifacts(_td, run_id=nested_run.info.run_id)

    flat_grad, _ = ravel_pytree(grad["laser"])
    loss = float(val)
    grad_norm = float(np.linalg.norm(flat_grad))

    mlflow.log_metrics({"loss": loss, "grad norm": grad_norm}, step=epoch)

    return val, flat_grad, grad["laser"]


def optax_loop(orig_cfg: Dict, modules: Dict):
    """
    Performs the optimization loop using optax.

    Args:
        parent_run_id: str: The parent run id
        orig_cfg: Dict: The original config
        modules: Dict: The modules


    """
    import optax
    import equinox as eqx

    lr_sched = optax.cosine_decay_schedule(
        init_value=orig_cfg["opt"]["learning_rate"],
        decay_steps=orig_cfg["opt"]["decay_steps"],
    )
    opt = optax.adam(learning_rate=lr_sched)
    opt_state = opt.init(eqx.filter(modules["laser"], eqx.is_array))  # initialize the optimizer state

    for i in range(20):  # range(200):
        _, _, laser_grad = calc_loss_and_grads(modules, i, orig_cfg)

        updates, opt_state = opt.update(laser_grad, opt_state, modules["laser"])
        modules["laser"] = eqx.apply_updates(modules["laser"], updates)


def scipy_loop(orig_cfg: Dict, modules: Dict) -> OptimizeResult:
    """
    Performs the optimization loop using scipy.

    The main reason this is different than the optax loop is because scipy prefers numpy arrays so
    the pytrees need to be flattened

    Args:
        parent_run_id: str: The parent run id
        orig_cfg: Dict: The original config
        modules: Dict: The modules

    Returns:
        result: The result of the optimization
    """
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree
    import numpy as np
    import equinox as eqx

    class Fitter:
        def __init__(self, _modules):
            self.model_cfg = _modules["laser"].model_cfg
            x0, self.static_params = eqx.partition(_modules["laser"], _modules["laser"].get_partition_spec())
            self.flattened_x0, self.unravel_pytree = ravel_pytree(x0)
            self.epoch = 0
            # self.parent_run_id = parent_run_id

        def loss_fn(self, flattened_x):
            diff_params = self.unravel_pytree(flattened_x)
            modules["laser"] = eqx.combine(diff_params, self.static_params)
            for k in self.model_cfg.keys():
                modules["laser"].model_cfg[k] = self.model_cfg[k]
            val, flat_grad, _ = calc_loss_and_grads(modules, self.epoch, orig_cfg)
            self.epoch += 1

            return float(val), np.array(flat_grad)  # jac=False -> return val only

        def fit(self):
            return minimize(
                self.loss_fn,
                np.array(self.flattened_x0, dtype=np.float32),
                jac=True,  # False, finite differencing
                method="L-BFGS-B",
                options={"maxiter": 20, "disp": True},
            )

    fitter = Fitter(modules)
    result = fitter.fit()

    return result


def run_opt(_cfg_path: str):
    """
    Sets up and runs the parent run which is the optimization loop

    Args:
        _cfg_path: str: Path to the config file


    """
    import uuid
    from copy import deepcopy
    from adept import ergoExo
    from adept import utils as adept_utils
    from ml_for_lpi.ml4tpd import TPDModule
    import jax

    logging.basicConfig(filename=f"runlog-tpd-learn-{str(uuid.uuid4())[-4:]}.log", level=logging.INFO)

    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    import yaml, mlflow, tempfile, os

    with open(_cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    if cfg["opt"]["method"] == "optax":
        optimization_loop = optax_loop
    elif cfg["opt"]["method"] == "scipy":
        optimization_loop = scipy_loop
    else:
        raise NotImplementedError(f"Optimization method {cfg['opt']['method']} not implemented.")

    _tt = cfg["units"]["reference electron temperature"]
    _gsl = cfg["density"]["gradient scale length"]
    _intensity = cfg["units"]["laser intensity"]

    # cfg["mlflow"]["run"] = f"temperature={_tt}-gsl={_gsl}-intensity={_intensity}"
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["mlflow"]["run"]) as mlflow_run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as td:
            with open(os.path.join(td, "config.yaml"), "w") as fi:
                yaml.dump(cfg, fi)
            mlflow.log_artifacts(td)
        # adept_utils.log_params(cfg)

        parent_run_id = mlflow_run.info.run_id
        orig_cfg = deepcopy(cfg)

    exo = ergoExo(mlflow_run_id=parent_run_id, mlflow_nested=False)
    modules = exo.setup(cfg, adept_module=TPDModule)

    with mlflow.start_run(run_id=parent_run_id, log_system_metrics=True) as mlflow_run:
        optimization_loop(orig_cfg, modules)

    return mlflow_run


if __name__ == "__main__":

    import argparse, mlflow

    parser = argparse.ArgumentParser(description="Run TPD training.")
    parser.add_argument("--cfg", type=str, help="The config file")
    parser.add_argument("--run_id", type=str, help="The run id")
    args = parser.parse_args()

    if args.run_id is not None:
        run_id = args.run_id
        cfg_path = os.path.join(mlflow.get_run(run_id).info.artifact_uri, "config.yaml")
    else:
        base_cfg_path = os.path.abspath(os.path.expanduser(args.cfg))
        cfg_path = f"{base_cfg_path}.yaml"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # exo = ergoExo()
    # with open(cfg_path, "r") as fi:
    #     cfg = yaml.safe_load(fi)
    # modules = exo.setup(cfg, TPDModule)
    # sol, post_out, run_id = exo.val_and_grad(modules)

    mlflow_run = run_opt(cfg_path)

# cfg_path = "/global/homes/n/ngub/adept/ml_for_lpi/configs/tpd_opt.yaml"
# run_opt(cfg_path)
