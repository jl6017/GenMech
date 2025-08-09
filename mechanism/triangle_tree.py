import numpy as np
import networkx as nx
from itertools import combinations
import networkx as nx
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import json


def node_types(G):
    """
    Assign node types: fixed, input, output, free
    """
    last_node = max(G.nodes())  # last node does not have a child node
    for node in G.nodes():
        if node == 0 or node == 2:
            G.nodes[node]['type'] = 'fixed'
        elif node == 1:
            G.nodes[node]['type'] = 'input'
        elif node == last_node:
            G.nodes[node]['type'] = 'output'
        else:
            G.nodes[node]['type'] = 'free'

    return G


def sequence_edges(edges):
    """
    Order the edges by the sequence of making the triangles
    """
    return sorted(edges, key=lambda x: (x[1], x[0]))  # sort by the second node, then by the first node


def generate_triangle_tree(num_triangles, filters=[False, False, False, False], save_figures=True):
    """
    structure search function for linkage, triangle layers from 2 to 5
    """
    record = []
    input_base = [
        (0, 1),
        (0, 2),
    ]
    G = nx.DiGraph()
    G.add_edges_from(input_base)
    G = node_types(G)
    graph_dict = {
        'tri_0': [G],
    }

    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "horizontalalignment": "center",
        "node_size": 500,
        "linewidths": 4,
        "width": 4,
        "arrowsize": 20,
        "arrowstyle": "-|>",
    }

    type_colors = {
        'fixed': 'grey',
        'input': 'red',
        'output': 'orange',
        'free': 'lightblue',
    }

    for i in range(1, num_triangles + 1):
        point_id = i + 2  # point_id starts from 3
        all_parents = list(combinations(range(point_id), 2))
        tri_i = []
        for p in all_parents:
            p1, p2 = sorted(p)

            tri_previous_G = graph_dict[f'tri_{i - 1}']
            for tri_g in tri_previous_G:
                new_link = list(tri_g.edges())
                new_link.append((p1, point_id))
                new_link.append((p2, point_id))

                # build a directed graph
                G = nx.DiGraph()
                G.add_edges_from(new_link)
                G_with_types = node_types(G)
                tri_i.append(G_with_types)
        graph_dict[f'tri_{i}'] = tri_i

    # record number of graphs of each triangle layer, after each filtering
    record.append([len(graphs) for graphs in graph_dict.values()])

    # filters
    if filters[0]:
        graph_dict = output_node_filter(graph_dict)
        record.append([len(graphs) for graphs in graph_dict.values()])
    if filters[1]:
        graph_dict = fixed_node_filter(graph_dict)
        record.append([len(graphs) for graphs in graph_dict.values()])
    if filters[2]:
        graph_dict = redundant_link_filter(graph_dict)
        record.append([len(graphs) for graphs in graph_dict.values()])
    if filters[3]:
        graph_dict = isomorph_filter(graph_dict)
        record.append([len(graphs) for graphs in graph_dict.values()])

    # save record as txt
    with open(f"mechanism/graph_record.txt", 'w') as f:
        # add header by graph_dict keys
        f.write("Graph Layer Record:\n")
        f.write(" -> ".join(graph_dict.keys()) + "\n")
        f.write("Number of graphs after each filter:\n")
        for i, r in enumerate(record):
            f.write(f"Filter {i}: {r}\n")


    # save figures
    filter_str = ''.join(['1' if f else '0' for f in filters])
    if save_figures:
        os.makedirs(f"mechanism/graph_vis/{filter_str}/", exist_ok=True)
        for key, graphs in graph_dict.items():
            for idx, g in tqdm(enumerate(graphs), total=len(graphs)):
                pos = nx.shell_layout(g)
                node_colors = [type_colors[g.nodes[n]['type']] for n in g.nodes()]
                options['node_color'] = node_colors
                plt.figure(figsize=(5, 5))
                nx.draw(g, pos, **options)
                plt.axis("off")
                plt.savefig(f"mechanism/graph_vis/{filter_str}/{key}_{idx:04d}.png")
                plt.close()

    # save graph dict to json
    graph_dict_json = {}
    for key, graphs in graph_dict.items():
        graph_dict_json[key] = []
        for idx, g in enumerate(graphs):
            edges = list(g.edges())
            sorted_edges = sequence_edges(edges)
            nodes = {n: g.nodes[n] for n in g.nodes()}
            graph_dict_json[key].append({
                'id': f"{key}_{idx:04d}",
                'edges': sorted_edges,
                'nodes': nodes
            })

    with open(f"mechanism/graph_vis/{filter_str}/graph_dict.json", 'w') as f:
        json.dump(graph_dict_json, f, indent=4)

    return graph_dict


def isomorph_filter(graph_dict):
    """
    only save isomorphic graphs, considering the node types
    """
    total_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print("Filtering isomorphic graphs...")
    node_match = lambda n1, n2: n1['type'] == n2['type']
    for key, graphs in graph_dict.items():
        unique_graphs_for_key = []
        for g in graphs:
            if not any(iso.is_isomorphic(g, ug, node_match=node_match) for ug in unique_graphs_for_key):
                unique_graphs_for_key.append(g)
        graph_dict[key] = unique_graphs_for_key
    
    after_filter_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print(f"{total_num_graphs} -> {after_filter_num_graphs}")

    return graph_dict


def output_node_filter(graph_dict):
    """
    only one output node, does not have a child node
    """
    total_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print("Filtering graphs with a single output node...")
    for key, graphs in graph_dict.items():
        filtered_graphs = []
        for g in graphs:
            no_child_nodes = [n for n in g.nodes() if g.out_degree(n) == 0]
            if len(no_child_nodes) == 1:
                # print(f"Graph {key} has a single output node: {no_child_nodes[0]}") # node id should be key - 1, the last node
                filtered_graphs.append(g)
            
        graph_dict[key] = filtered_graphs
    after_filter_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print(f"{total_num_graphs} -> {after_filter_num_graphs}")

    return graph_dict


def redundant_link_filter(graph_dict):
    """
    remove redundant links, for mesh triangles: 
    - two triangles share one same edge, this is a rigid body
    - the triangles connected to the two fixed nodes, this introduces another fixed node
    1. record the mesh triangles as a list of three nodes
    2. if there exists two triangles with the same two nodes, this two triangles are connected
    """
    total_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print("Filtering redundant links...")
    for key, graphs in graph_dict.items():
        filtered_graphs = []
        for g in graphs:
            # Record the rigid triangles
            rigid_triangles = []
            for node in g.nodes():
                parents = list(g.predecessors(node))
                if len(parents) == 2:
                    p1, p2 = sorted(parents)  # in our case, always small to large
                    if g.has_edge(p1, p2):
                        rigid_triangles.append((p1, p2, node))

            # check if triangles are connected
            add_graph = True
            if len(rigid_triangles) > 1:
                for t1, t2 in combinations(rigid_triangles, 2):
                    if len(set(t1) & set(t2)) == 2:
                        # if two triangles share two nodes, they are connected
                        add_graph = False
                        break

            if add_graph:
                filtered_graphs.append(g)

        graph_dict[key] = filtered_graphs
    after_filter_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print(f"{total_num_graphs} -> {after_filter_num_graphs}")

    return graph_dict


def fixed_node_filter(graph_dict):
    """
    remove the graphs that have nodes connected to 2 fixed nodes
    remove the graphs that output node is connected to a fixed node, avoid output node drawing circle or arc
    """
    total_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print("Filtering graphs with fixed nodes conditions...")
    for key, graphs in graph_dict.items():
        filtered_graphs = []
        for g in graphs:
            add_graph = True
            for node in g.nodes():
                # check if the node is connected to two fixed nodes
                parents = list(g.predecessors(node))
                if len(parents) == 2 and all(g.nodes[p]['type'] == 'fixed' for p in parents):
                    add_graph = False
                    break
                # check if the output node is connected to a fixed node
                if g.nodes[node]['type'] == 'output':
                    if any(g.nodes[p]['type'] == 'fixed' for p in parents):
                        add_graph = False
                        break

            if add_graph:
                filtered_graphs.append(g)

        graph_dict[key] = filtered_graphs
    after_filter_num_graphs = sum(len(graphs) for graphs in graph_dict.values())
    print(f"{total_num_graphs} -> {after_filter_num_graphs}")

    return graph_dict


if __name__ == "__main__":
    num_triangles = 5
    triangle_graph = generate_triangle_tree(num_triangles, filters=[True, True, True, True], save_figures=True)
