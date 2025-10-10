import argparse
import pathlib
import pickle
from typing import Optional, Sequence

import numpy as np

from graph_utils import read_sndlib_topology, read_txt_file

from optical_rl_gym.utils import (
    Span,
    Link,
    Path,
    Modulation,
    get_best_modulation_format,
    get_k_shortest_paths,
    get_path_weight,
)

# from optical_rl_gym.core import (
#     Span,
#     Link,
#     Path,
# )

max_span_length = 80  # km
default_attenuation = 0.2  # dB/km  ==> 0.3 dB/km
# TODO: from 4.5 to 5.5 dB
default_noise_figure = 4.5  # 10 ** (5 / 10)  # dB ===> norm  # can be from 4.5-5.5

# in case you do not have modulations
modulations: Optional[Sequence[Modulation]] = None

# note that minimum OSNR and in-band cross-talk are optional parameters

# example of in-band cross-talk settings for different modulation formats:
# https://ieeexplore.ieee.org/abstract/document/7541954
# table III

# defining the EON parameters
# definitions according to:
# https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/K-SP-FF%20benchmark_NSFNET.py#L268
# modulations = (
#     # the first (lowest efficiency) modulation format needs to have maximum length
#     # greater or equal to the longest path in the topology.
#     # Here we put 100,000 km to be on the safe side
#     Modulation(
#         name="BPSK", maximum_length=100_000, spectral_efficiency=1, minimum_osnr=12.6, inband_xt=-14
#     ),
#     Modulation(
#         name="QPSK", maximum_length=2_000, spectral_efficiency=2, minimum_osnr=12.6, inband_xt=-17
#     ),
#     Modulation(
#         name="8QAM", maximum_length=1_250, spectral_efficiency=3, minimum_osnr=18.6, inband_xt=-20
#     ),
#     Modulation(
#         name="16QAM", maximum_length=625, spectral_efficiency=4, minimum_osnr=22.4, inband_xt=-23
#     ),
# )

# values for HD-FEC:
# target_SNR_dB_vec = [5.51778473797453	8.52808469461434	12.5122814809365	15.1925724496781	18.1907513965596	21.1216830862200]; % in dB

# values for SD-FEC:
target_SNR_dB_vec = (3.71925646843142, 6.72955642507124, 10.8453935345953, 13.2406469649752, 16.1608982942870, 19.0134649345090)

# other setup:
modulations = (
    # the first (lowest efficiency) modulation format needs to have maximum length
    # greater or equal to the longest path in the topology.
    # Here we put 100,000 km to be on the safe side
    Modulation(
        name="BPSK",
        maximum_length=100_000,
        spectral_efficiency=1,
        minimum_osnr=target_SNR_dB_vec[0],
    ),
    Modulation(
        name="QPSK",
        maximum_length=2_000,
        spectral_efficiency=2,
        minimum_osnr=target_SNR_dB_vec[1],
    ),
    Modulation(
        name="8QAM",
        maximum_length=1_000,
        spectral_efficiency=3,
        minimum_osnr=target_SNR_dB_vec[2],
    ),
    Modulation(
        name="16QAM",
        maximum_length=500,
        spectral_efficiency=4,
        minimum_osnr=target_SNR_dB_vec[3],
    ),
    Modulation(
        name="32QAM",
        maximum_length=250,
        spectral_efficiency=5,
        minimum_osnr=target_SNR_dB_vec[4],
    ),
    Modulation(
        name="64QAM",
        maximum_length=125,
        spectral_efficiency=6,
        minimum_osnr=target_SNR_dB_vec[5],
    ),
)


def findSpanLen(link_length, max_span_len):
    #print(total, divisor)
    quotient = link_length // max_span_len  # How many times divisor fits fully into total
    remainder = link_length % max_span_len  # Remaining part after full divisions
    #print(quotient,divisor)
    result = [max_span_len] * int(quotient)  # List of full divisions
    if remainder > 0:
        result.append(remainder)  # Add the remainder as the last element
    return result

def get_topology(file_name, topology_name, modulations, k_paths=5):
    k_shortest_paths = {}
    max_length = 0
    min_length = 1e12
    if file_name.endswith(".xml"):
        topology = read_sndlib_topology(file_name)
    elif file_name.endswith(".txt"):
        topology = read_txt_file(file_name)
    else:
        raise ValueError("Supplied topology is unknown")

    for node1, node2 in topology.edges():
        length = topology[node1][node2]["length"]
        num_spans = int(length // max_span_length) + 1
        print(f"{num_spans=}")
        #span_length = length / num_spans
        span_lengths = findSpanLen(length, max_span_length)
        print(len(span_lengths))
        spans = []
        for i in range(len(span_lengths)):
            span = Span(
                length=span_lengths[i],
                #attenuation=default_attenuation,
                #noise_figure=default_noise_figure,
            )
            spans.append(span)

        link = Link(
            id=topology[node1][node2]["index"],
            length=length,
            node1=node1,
            node2=node2,
            spans=tuple(spans),
        )
        topology[node1][node2]["link"] = link
        print(link)

    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 < idn2:
                paths = get_k_shortest_paths(topology, n1, n2, k_paths, weight="length")
                print(n1, n2, len(paths))
                lengths = [
                    get_path_weight(topology, path, weight="length") for path in paths
                ]
                objs = []

                for path, length in zip(paths, lengths):
                    links = []
                    for i in range(len(path) - 1):
                        link = topology[path[i]][path[i+1]]["link"]
                        links.append(link)
                    objs.append(
                        Path(
                            path_id=idp,
                            hops=len(path) - 1,
                            length=length,
                            node_list=tuple(path),
                            # best_modulation=modulation,
                            links=tuple(links),
                        )
                    )  # <== The topology is created and a best modulation is just automatically attached.  In our new implementation, the best modulation will be variable depending on available resources and the amount of crosstalk it will cause.
                    print("\t", objs[-1])
                    idp += 1
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)
                k_shortest_paths[n1, n2] = objs
                k_shortest_paths[n2, n1] = objs
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    if modulations is not None:
        topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx
    print("Max length:", max_length)
    print("Min length:", min_length)
    return topology


if __name__ == "__main__":
    # default values
    k_paths = 5
    topology_file = "nsfnet_chen.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--k_paths",
        type=int,
        default=k_paths,
        help="Number of k-shortest-paths to be considered (default={})".format(k_paths),
    )
    parser.add_argument(
        "-t",
        "--topology",
        default=topology_file,
        help="Network topology file to be used. Default: {}".format(topology_file),
    )

    args = parser.parse_args()

    topology_path = pathlib.Path(args.topology)

    topology = get_topology(
        args.topology, topology_path.stem.upper(), modulations, args.k_paths
    )

    import time
    total = 0
    for i in range(50):
        start = time.time_ns()
        s = 0
        for paths in topology.graph["ksp"].values():
            for path in paths:
                for link in path.links:
                    for span in link.spans:
                        s += span.length
        end = time.time_ns()
        total += (end - start) / 1e6
    print(total)

    file_name = topology_path.stem + "_" + str(k_paths) + "-paths"
    if modulations is not None:
        file_name += "_" + str(len(modulations)) + "-modulations"
    file_name += ".h5"

    file_name = topology_path.stem + "_link_span_" + str(k_paths) + "-paths"
    if modulations is not None:
        file_name += "_" + str(len(modulations)) + "-modulations"
    file_name += ".h5"

    output_file = topology_path.parent.resolve().joinpath("examples").joinpath("topologies").joinpath(file_name)
    with open(file_name, "wb") as f:
        pickle.dump(topology, f)

    print("done for", topology)
