import argparse
import pathlib
import pickle
from typing import Optional, Sequence
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Union


import argparse
import pathlib
import pickle
from typing import Optional, Sequence
from itertools import islice
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


# Constants
max_span_length = 80  # km
default_attenuation = 0.2  # dB/km
default_noise_figure = 4.5  # dB

max_num_spans = 0

# Target SNR values for SD-FEC
target_SNR_dB_vec = (3.71925646843142, 6.72955642507124, 10.8453935345953, 
                     13.2406469649752, 16.1608982942870, 19.0134649345090)



def read_txt_to_digraph(file_path: str) -> nx.DiGraph:
    """
    Read topology from txt file and create a directed graph.
    Handles bidirectional links with proper indices.
    """
    graph = nx.DiGraph()
    num_nodes = 0
    num_links = 0
    id_link = 0
    
    with open(file_path, "r") as lines:
        # gets only lines that do not start with the # character
        nodes_lines = [value for value in lines if not value.startswith("#")]
        
        for idx, line in enumerate(nodes_lines):
            if idx == 0:
                num_nodes = int(line)
                # Add nodes with string IDs
                for id in range(1, num_nodes + 1):
                    graph.add_node(str(id), name=str(id))
                    
            elif idx == 1:
                num_links = int(line)
                
            elif len(line) > 1:
                info = line.replace("\n", "").split(" ")
                # Add forward edge
                graph.add_edge(
                    info[0],
                    info[1],
                    id=id_link,
                    index=id_link,
                    weight=1,
                    length=int(info[2]),
                )
                id_link += 1
                
                # Add reverse edge
                graph.add_edge(
                    info[1],
                    info[0],
                    id=id_link,
                    index=id_link,
                    weight=1,
                    length=int(info[2]),
                )
                id_link += 1
                
    return graph

def get_k_shortest_paths(topology: nx.DiGraph, source: int, target: int, k: int, weight: str = 'length') -> List:
    """Get k-shortest paths using networkx"""
    #return list(nx.shortest_simple_paths(topology, source, target, weight=weight))[:k]
    return list(islice(nx.shortest_simple_paths(topology, source, target, weight=weight), k))

def get_path_weight(topology: nx.DiGraph, path: List, weight: str = 'length') -> float:
    """Calculate total path weight"""
    return sum(topology[path[i]][path[i+1]][weight] for i in range(len(path)-1))

# def get_topology(file_name: str, topology_name: str, modulations: Sequence[Modulation], k_paths: int = 5) -> nx.DiGraph:
#     """
#     Create topology with k-shortest paths and physical parameters.
#     """
#     k_shortest_paths = {}
#     max_length = 0
#     min_length = 1e12

#     # Create DiGraph directly
#     topology = read_txt_to_digraph(file_name)
    
#     # Add spans and links
#     for node1, node2 in topology.edges():
#         length = topology[node1][node2]["length"]
#         #num_spans = int(length // max_span_length) + 1
#         num_spans = int(length // 80) + (1 if length % 80 != 0 else 0)
#         if (num_spans >= max_num_spans):
#             max_num_spans=num_spans
#         # Create spans
#         spans = []
#         remaining_length = length

#         # Add full 80-length spans
#         for i in range(num_spans - 1):
#             spans.append(Span(length=80))
#             remaining_length -= 80

#         # Add the last span with remaining length
#         if remaining_length > 0:
#             spans.append(Span(length=remaining_length))

#         print(spans)

#         link = Link(
#             id=topology[node1][node2]["index"],
#             length=length,
#             node1=node1,
#             node2=node2,
#             spans=tuple(spans)
#         )
#         topology[node1][node2]["link"] = link

#     # Calculate k-shortest paths
#     # Calculate k-shortest paths
#     idp = 0
#     for idn1, n1 in enumerate(topology.nodes()):
#         for idn2, n2 in enumerate(topology.nodes()):
#             if idn1 != idn2:  # Process all pairs except self-loops
#                 # Get forward paths (n1 -> n2)
#                 forward_paths = get_k_shortest_paths(topology, n1, n2, k_paths)
#                 forward_lengths = [get_path_weight(topology, path) for path in forward_paths]
#                 forward_objs = []

#                 for path, length in zip(forward_paths, forward_lengths):
#                     links = []
#                     link_idx = []
#                     for i in range(len(path) - 1):
#                         link = topology[path[i]][path[i+1]]["link"]
#                         link_index = topology[path[i]][path[i+1]]["index"]
#                         links.append(link)
#                         link_idx.append(link_index)
                    
#                     obj = Path(
#                         path_id=idp,
#                         hops=len(path) - 1,
#                         length=length,
#                         node_list=tuple(path),
#                         links=tuple(links),
#                         link_idx=link_idx
#                     )
#                     forward_objs.append(obj)
#                     print(f"\tForward {obj}")
#                     idp += 1
#                     max_length = max(max_length, length)
#                     min_length = min(min_length, length)

#                 # Get reverse paths (n2 -> n1)
#                 reverse_paths = get_k_shortest_paths(topology, n2, n1, k_paths)
#                 reverse_lengths = [get_path_weight(topology, path) for path in reverse_paths]
#                 reverse_objs = []

#                 for path, length in zip(reverse_paths, reverse_lengths):
#                     links = []
#                     link_idx = []
#                     for i in range(len(path) - 1):
#                         link = topology[path[i]][path[i+1]]["link"]
#                         link_index = topology[path[i]][path[i+1]]["index"]
#                         links.append(link)
#                         link_idx.append(link_index)
                    
#                     obj = Path(
#                         path_id=idp,
#                         hops=len(path) - 1,
#                         length=length,
#                         node_list=tuple(path),
#                         links=tuple(links),
#                         link_idx=link_idx
#                     )
#                     reverse_objs.append(obj)
#                     print(f"\tReverse {obj}")
#                     idp += 1
#                     max_length = max(max_length, length)
#                     min_length = min(min_length, length)

#                 # Store forward and reverse paths separately
#                 k_shortest_paths[n1, n2] = forward_objs
#                 k_shortest_paths[n2, n1] = reverse_objs

#     # Add graph attributes
#     topology.graph["name"] = topology_name
#     topology.graph["ksp"] = k_shortest_paths
#     topology.graph["modulations"] = modulations
#     topology.graph["k_paths"] = k_paths
#     topology.graph["node_indices"] = []
#     topology.graph['max_path_length'] = max_length
    
#     # Add node indices
#     for idx, node in enumerate(topology.nodes()):
#         topology.graph["node_indices"].append(node)
#         topology.nodes[node]["index"] = idx

#     print(f"Max length: {max_length}")
#     print(f"Min length: {min_length}")
#     return topology

def get_topology(file_name: str, topology_name: str, modulations: Sequence[Modulation], k_paths: int = 5) -> nx.DiGraph:
    """
    Create topology with k-shortest paths and physical parameters.
    Also computes span counts per path and global metrics using the longest
    path among the k-shortest paths for each (s,d).
    """
    k_shortest_paths = {}
    max_length = 0
    min_length = 1e12

    # --- New accumulators ---
    longest_paths_set = []   # store the longest path per (s,d)
    max_num_links = 0
    max_num_spans = 0
    total_link_length = 0.0
    total_links_count = 0
    total_path_length = 0.0
    total_spans = 0
    path_count = 0
    total_k_paths = 0
    longest_path = None

    # Create DiGraph directly
    topology = read_txt_to_digraph(file_name)
    
    # Add spans and links
    for node1, node2 in topology.edges():
        length = topology[node1][node2]["length"]
        num_spans = int(length // max_span_length) + (1 if length % max_span_length != 0 else 0)

        spans = []
        remaining_length = length
        for i in range(num_spans - 1):
            spans.append(Span(length=max_span_length))
            remaining_length -= max_span_length
        if remaining_length > 0:
            spans.append(Span(length=remaining_length))

        link = Link(
            id=topology[node1][node2]["index"],
            length=length,
            node1=node1,
            node2=node2,
            spans=tuple(spans)
        )
        topology[node1][node2]["link"] = link

    # Calculate k-shortest paths
    idp = 0
    for idn1, n1 in enumerate(topology.nodes()):
        for idn2, n2 in enumerate(topology.nodes()):
            if idn1 != idn2:  # Process all pairs except self-loops
                # Get forward paths (n1 -> n2)
                forward_paths = get_k_shortest_paths(topology, n1, n2, k_paths)
                forward_lengths = [get_path_weight(topology, path) for path in forward_paths]
                forward_objs = []

                pair_longest_path = None
                pair_longest_spans = 0

                for path, length in zip(forward_paths, forward_lengths):
                    links = []
                    link_idx = []
                    num_spans_path = 0
                    for i in range(len(path) - 1):
                        link = topology[path[i]][path[i+1]]["link"]
                        link_index = topology[path[i]][path[i+1]]["index"]
                        links.append(link)
                        link_idx.append(link_index)
                        num_spans_path += len(link.spans)

                    obj = Path(
                        path_id=idp,
                        hops=len(path) - 1,
                        length=length,
                        node_list=tuple(path),
                        links=tuple(links),
                        link_idx=link_idx,
                        num_spans=num_spans_path
                    )
                    #obj.num_spans = num_spans_path  # NEW: store spans count
                    forward_objs.append(obj)
                    print(f"\tForward {obj}")
                    idp += 1
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)

                    # track longest among k paths
                    if pair_longest_path is None or length > pair_longest_path.length:
                        pair_longest_path = obj
                        pair_longest_spans = num_spans_path

                # Get reverse paths (n2 -> n1)
                reverse_paths = get_k_shortest_paths(topology, n2, n1, k_paths)
                reverse_lengths = [get_path_weight(topology, path) for path in reverse_paths]
                reverse_objs = []

                pair_longest_rev = None
                pair_longest_rev_spans = 0

                for path, length in zip(reverse_paths, reverse_lengths):
                    links = []
                    link_idx = []
                    num_spans_path = 0
                    for i in range(len(path) - 1):
                        link = topology[path[i]][path[i+1]]["link"]
                        link_index = topology[path[i]][path[i+1]]["index"]
                        links.append(link)
                        link_idx.append(link_index)
                        num_spans_path += len(link.spans)

                    obj = Path(
                        path_id=idp,
                        hops=len(path) - 1,
                        length=length,
                        node_list=tuple(path),
                        links=tuple(links),
                        link_idx=link_idx,
                        num_spans=num_spans_path
                    )
                    #obj.num_spans = num_spans_path  # NEW
                    reverse_objs.append(obj)
                    print(f"\tReverse {obj}")
                    idp += 1
                    max_length = max(max_length, length)
                    min_length = min(min_length, length)

                    # track longest among k paths
                    if pair_longest_rev is None or length > pair_longest_rev.length:
                        pair_longest_rev = obj
                        pair_longest_rev_spans = num_spans_path

                # Store forward and reverse paths separately
                k_shortest_paths[n1, n2] = forward_objs
                k_shortest_paths[n2, n1] = reverse_objs

                # Add longest per pair to set
                if pair_longest_path:
                    longest_paths_set.append((pair_longest_path, pair_longest_spans))
                if pair_longest_rev:
                    longest_paths_set.append((pair_longest_rev, pair_longest_rev_spans))

                total_k_paths += len(forward_paths) + len(reverse_paths)

    # --- Compute new metrics from longest paths ---
    for path_obj, spans in longest_paths_set:
        if path_obj.length > max_length:
            longest_path = path_obj
        max_num_links = max(max_num_links, path_obj.hops)
        max_num_spans = max(max_num_spans, spans)

        # avg link length
        for link in path_obj.links:
            total_link_length += link.length
            total_links_count += 1

        total_path_length += path_obj.length
        total_spans += spans
        path_count += 1

    avg_link_length = total_link_length / total_links_count if total_links_count > 0 else 0.0
    avg_path_length = total_path_length / path_count if path_count > 0 else 0.0
    avg_spans_per_path = total_spans / path_count if path_count > 0 else 0.0

    # Add graph attributes
    topology.graph["name"] = topology_name
    topology.graph["ksp"] = k_shortest_paths
    topology.graph["modulations"] = modulations
    topology.graph["k_paths"] = k_paths
    topology.graph["node_indices"] = []
    topology.graph['max_path_length'] = max_length
    topology.graph['max_num_links'] = max_num_links
    topology.graph['max_num_spans'] = max_num_spans
    topology.graph['avg_link_length'] = avg_link_length
    topology.graph['avg_path_length'] = avg_path_length
    topology.graph['avg_spans_per_path'] = avg_spans_per_path
    topology.graph['total_k_paths'] = total_k_paths
    topology.graph['total_longest_paths'] = len(longest_paths_set)
    if longest_path:
        topology.graph["longest_path"] = {
            "path_id": longest_path.path_id,
            "nodes": longest_path.node_list,
            "length": longest_path.length,
            "hops": longest_path.hops,
            "spans": longest_path.num_spans,
        }

    # Add node indices
    for idx, node in enumerate(topology.nodes()):
        topology.graph["node_indices"].append(node)
        topology.nodes[node]["index"] = idx

    # Print stats
    print("========== Topology Metrics ==========")
    print(f"Total k-shortest paths : {total_k_paths}")
    print(f"Total longest paths    : {len(longest_paths_set)} (one per (s,d) pair, both directions)")
    print(f"Max path length        : {max_length:.2f} km")
    print(f"Max number of links    : {max_num_links}")
    print(f"Max number of spans    : {max_num_spans}")
    print(f"Average link length    : {avg_link_length:.2f} km")
    print(f"Average path length    : {avg_path_length:.2f} km")
    print(f"Average spans per path : {avg_spans_per_path:.2f}")
    if longest_path:
        print(f"Longest path           : ID={longest_path.path_id}, "
              f"nodes={longest_path.node_list}, "
              f"length={longest_path.length:.2f} km, "
              f"hops={longest_path.hops}, "
              f"spans={longest_path.num_spans}")
    print("======================================")

    print(f"Max length: {max_length}")
    print(f"Min length: {min_length}")
    return topology




if __name__ == "__main__":

     # default values
    k_paths = 5
    topology_file = "indian_net.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        "--k_paths",
        type=int,
        default=5,
        help="Number of k-shortest-paths to consider"
    )
    parser.add_argument(
        "-t",
        "--topology",
        default="indian_net.txt",
        help="Network topology file"
    )

    args = parser.parse_args()
    topology_path = pathlib.Path(args.topology)

    # Create topology
    topology = get_topology(
        args.topology, 
        topology_path.stem.upper(), 
        args.k_paths
    )

    # Save topology
    file_name = f"{topology_path.stem}_{args.k_paths}-paths_new.h5"
    output_path = topology_path.parent.resolve().joinpath(file_name)
    
    with open(output_path, "wb") as f:
        pickle.dump(topology, f)

    print(f"Topology saved to {output_path}")