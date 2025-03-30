from typing import Optional, Dict
from e3nn.o3 import Irreps


class GraphModuleIrreps:
    """Class deeply inspired and partially copied from NequiPs GraphModuleMixin which keeps track of the different irreps going and out a module of the graph neural networks
    Here, we use a highly diluted version of it.
    """

    def _init_irreps(
        self,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        irreps_out: Optional[Dict[str, Irreps]] = {},
    ):
        """_summary_

        Args:
            irreps_in (Optional[Dict[str, Irreps]], optional): _description_. Defaults to {}.
            irreps_out (Optional[[Dict[str, Irreps]]], optional): _description_. Defaults to {}.
        """
        self.irreps_in = irreps_in
        # outputs are just the irreps in updated with the given outputs
        # taken from NequiP.nn._graph_mixin.GraphModuleMixin
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out
