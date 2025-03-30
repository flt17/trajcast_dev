import os
from typing import Annotated

import torch
import typer

from trajcast.model.models import EfficientTrajCastModel

app = typer.Typer()

CONFIG_MSG = "Path to model config."
STATE_DICT_MSG = "Path to state dict with weights."
OUT_NAME_MSG = "Name of model or state dict with converted o3 backend."
DEVICE_MSG = "Device to be used here for conversion."
WRITE_MSG = "Whether to write the state dict and config file to disk."


def transfer_weights(
    source_model: EfficientTrajCastModel,
    target_model: EfficientTrajCastModel,
):
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    # first transfer common keys
    shared_keys = set(source_state_dict.keys()) & set(target_state_dict.keys())
    for key in shared_keys:
        target_state_dict[key] = source_state_dict[key]

    return target_state_dict


def _check_dtypes(state_dict: dict):

    dtypes = {param.dtype for param in state_dict.values() if param.is_floating_point()}
    if len(dtypes) > 1:
        raise TypeError(f"Mixed precision in model:{dtypes}")
    else:
        dtype = dtypes.pop()
        torch.set_default_dtype(dtype)


def convert_state_dicts(
    source_model: EfficientTrajCastModel,
    device: str,
    write: bool = True,
    out_name: str = None,
):
    target_model = source_model.__class__(config=source_model.config)

    source_backend = source_model.o3_backend
    target_backend = {"e3nn", "cueq"} ^ {source_backend}

    target_model.o3_backend = target_backend.pop()

    # transfer weights from source to target
    target_state_dict = transfer_weights(
        source_model,
        target_model,
    )
    if write:
        output_state_dict = f"state_dict_{out_name}.pt"
        output_yaml = f"config_{out_name}.yaml"

        torch.save(target_state_dict, output_state_dict)
        target_model.load_state_dict(torch.load(output_state_dict, map_location=device))
        target_model.dump_config_to_yaml(output_yaml)

    return target_state_dict


@app.command()
def main(
    config: Annotated[str, typer.Argument(help=CONFIG_MSG)],
    state_dict: Annotated[str, typer.Argument(help=STATE_DICT_MSG)],
    write: Annotated[bool, typer.Option(help=WRITE_MSG)] = True,
    out_name: Annotated[
        str,
        typer.Option(
            help=OUT_NAME_MSG,
        ),
    ] = "e3nn",
    device: Annotated[str, typer.Option(help=DEVICE_MSG)] = "cpu",
):
    if not os.path.exists(config):
        raise FileNotFoundError(f"Model config not found under given path:{config}!")

    if not os.path.exists(state_dict):
        raise FileNotFoundError(f"State dict not found under given path:{state_dict}!")

    # start by loading the state dict
    state = torch.load(state_dict, map_location=device)

    # check the precision
    _check_dtypes(state)

    # instantiate model
    model = EfficientTrajCastModel.build_from_yaml(config)

    # we do not check whether state dict and config are compatible here
    model.load_state_dict(state)

    convert_state_dicts(
        source_model=model, device=device, write=write, out_name=out_name
    )


if __name__ == "__main__":
    app()
