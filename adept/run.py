import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config, jit, value_and_grad
import jax.numpy as jnp

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

import yaml

from adept import ergoExo
import sys

ml_for_lpi_path = "/global/homes/n/ngub/adept"
sys.path.append(os.path.abspath(ml_for_lpi_path))
# from ml_for_lpi.ml4tpd import TPDModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Differentiation Enabled Plasma Transport")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--run_id", help="enter run_id to continue")
    args = parser.parse_args()

    exo = ergoExo()

    if args.run_id is None:

        base_cfg_path = os.path.abspath(os.path.expanduser(args.cfg))
        config_file_path = f"{base_cfg_path}.yaml"
        with open(config_file_path, "r") as fi:
            cfg = yaml.safe_load(fi)

        modules = exo.setup(cfg=cfg)
        sol, post_out, run_id = exo(modules)
        
        # modules = exo.setup(cfg, TPDModule)
        # sol, post_out, run_id = exo.val_and_grad(modules)

    else:
        exo.run_job(args.run_id, nested=None)
        run_id = args.run_id