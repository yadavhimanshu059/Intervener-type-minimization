# Using NetworkX package and conllu package
import os
from io import open
from conllu import parse
import networkx as nx
import matplotlib.pyplot as plt
from Measures import Compute_measures  # Corrected class name

directory = "./SUD"  # directory containing the UD scheme tree files in CONLLU format
output_directory = "./output"  # directory to save the output histogram
os.makedirs(output_directory, exist_ok=True)  # create output directory if it doesn't exist

ud_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('train.conllu'):
            fullpath = os.path.join(root, file)
            ud_files.append(fullpath)  # creates a list of path of all files (file of each language) from the directory

# Dictionary to accumulate intervener POS tags
intervener_pos_tags = {}

# Limit the number of sentences to process
max_sentences = 50

for i in ud_files:  # reads file of each language one by one
    lang = str(i).replace("./SUD", "")
    lang = lang.replace("-sud-train.conllu", "")  # lang variable stores the language code
    lang = lang.replace("-sud-test.conllu", "")
    data_file = open(str(i), 'r', encoding='utf-8').read()
    sentences = parse(data_file)  # parses the CONLLU format
    sent_id = 0
    print(lang)
    num_sent = 0
    num_edge = 0

    for sentence in sentences[:max_sentences]:
        sent_id += 1
        print(f"Processing sentence {sent_id} of {lang}")
        if sent_id > 1:
            num_sent += 1
            tree = nx.DiGraph()  # An empty directed graph (i.e., edges are uni-directional)
            for nodeinfo in sentence:  # retrieves information of each node from dependency tree in UD format
                entry = list(nodeinfo.items())
                if entry[7][1] != 'punct':
                    tree.add_node(entry[0][1], form=entry[1][1], lemma=entry[2][1], upostag=entry[3][1], xpostag=entry[4][1], feats=entry[5][1], head=entry[6][1], deprel=entry[7][1], deps=entry[8][1], misc=entry[9][1])  # adds node to the directed graph
            ROOT = 0
            tree.add_node(ROOT)  # adds an abstract root node to the directed graph

            for nodex in tree.nodes:
                if nodex != 0:
                    if tree.has_node(tree.nodes[nodex]['head']):  # to handle disjoint trees
                        tree.add_edge(tree.nodes[nodex]['head'], nodex, drel=tree.nodes[nodex]['deprel'])  # adds edges as relation between nodes

            n = len(tree.edges)
            if 1 < n < 12:
                get = Compute_measures(tree)
                # Computes the measures for the real tree
                projection_degree_real = get.projection_degree(0)  # gives the projection degree of the tree i.e., size of longest projection chain in the tree
                for edgey in tree.edges:
                    if edgey[0] != 0:
                        direction_real = get.dependency_direction(edgey)  # direction of the edge in terms of relative linear order of head and its dependent
                        dep_distance_real = get.dependency_distance(edgey)  # gives the distance between nodes connected by an edge
                        dep_depth_real = get.dependency_depth(edgey)
                        head_pos = tree.nodes[edgey[0]]['upostag']
                        dependent_pos = tree.nodes[edgey[1]]['upostag']
                        dependency_relation = tree[edgey[0]][edgey[1]]['drel']
                        dep_intervener_real = get.intervener_pos(edgey)
                        dep_relation_real = get.relation_pos(edgey)
                        dep_head_real = get.head_pos(edgey)
                        print("head_pos: {}, dependent_pos: {}, dependency_relation: {}".format(head_pos, dependent_pos, dependency_relation))  # Debugging print

                        # Accumulate intervener POS tags
                        for pos_tag in dep_intervener_real:
                            if pos_tag in intervener_pos_tags:
                                intervener_pos_tags[pos_tag] += 1
                            else:
                                intervener_pos_tags[pos_tag] = 1

                        with open('English-measures.csv', 'a') as results2:
                            results2.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                lang, "real", sent_id, n, projection_degree_real, edgey, direction_real, dep_distance_real, dep_depth_real, head_pos, dependent_pos, dependency_relation, dep_intervener_real, dep_head_real,
                            ))

                    print("\n-----------------\n" + str(tree.edges))

# Generate and save the histogram of intervener POS tags
plt.figure(figsize=(10, 6))
plt.bar(intervener_pos_tags.keys(), intervener_pos_tags.values())
plt.xlabel('Part of Speech')
plt.ylabel('Frequency')
plt.title('Distribution of Intervener POS Tags')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('intervener_pos_tags_histogram.png')
plt.show()
