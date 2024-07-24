import os
from io import open
from conllu import parse
import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')  # Replace 'Qt5Agg' with another backend like 'TkAgg', 'Agg', etc.
import matplotlib.pyplot as plt
import PyQt5  # or import PySide2
from wordcloud import WordCloud  # Ensure this import is present and correct
import seaborn as sns
import pandas as pd
from Measures import *
from Measures_rand import *
import random
from baseline_conditions_DL_matched_RLAs import *

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
intervener_pos_tags_real = {}
intervener_pos_tags_rand = {}

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

    for sentence in sentences[:max_sentences]:
        sent_id += 1
        print(f"Processing sentence {sent_id} of {lang}")

        if sent_id > 1:
            num_sent += 1
            tree = nx.DiGraph()  # An empty directed graph (i.e., edges are uni-directional)

            for nodeinfo in sentence:
                entry = list(nodeinfo.items())
                if entry[7][1] != 'punct':
                    tree.add_node(entry[0][1], form=entry[1][1], lemma=entry[2][1], upostag=entry[3][1], xpostag=entry[4][1],
                                  feats=entry[5][1], head=entry[6][1], deprel=entry[7][1], deps=entry[8][1], misc=entry[9][1])

            ROOT = 0
            tree.add_node(ROOT)  # adds an abstract root node to the directed graph

            for nodex in tree.nodes:
                if nodex != 0:
                    if tree.has_node(tree.nodes[nodex]['head']):  # to handle disjoint trees
                        tree.add_edge(tree.nodes[nodex]['head'], nodex, drel=tree.nodes[nodex]['deprel'])  # adds edges as relation between nodes

            n = len(tree.edges)
            if 1 < n < 12:
                get = Compute_measures(tree)
                num_cross_real = 0
                for edgex in tree.edges:
                    if edgex[0] != ROOT:
                        if get.is_projective(edgex):  # checks if edge is projective or not
                            num_cross_real += 0
                        else:
                            num_cross_real += 1

                generate = Random_base(tree)  # initiates object for computing measures for the real tree
                ls_random = generate.gen_random(num_cross_real)  # stores the list of random generated trees

                if ls_random:
                    treex = ls_random[0]
                    root = 1000
                    find = Compute_measures_rand(treex, root)  # initiates object for computing measures for the random tree
                    max_arity_rand = find.arity()[0]  # gives maximum arity present in the tree
                    avg_arity_rand = find.arity()[1]
                    projection_degree_rand = find.projection_degree(root)  # gives the projection degree of the tree i.e., size of longest projection chain in the tree
                    gap_degree_rand = find.gap_degree(root)  # gives gap_degree of the tree
                    k_illnest_rand = find.illnestedness(root, gap_degree_rand)

                    for edgex in treex.edges:
                        if edgex[0] != root and edgex[0] in tree.nodes and edgex[1] in tree.nodes:
                            direction_rand = get.dependency_direction(edgex)  # direction of the edge in terms of relative linear order of head and its dependent
                            dep_distance_rand = get.dependency_distance(edgex)  # gives the distance between nodes connected by an edge
                            dep_depth_rand = get.dependency_depth(edgex)
                            head_pos = tree.nodes[edgex[0]]['upostag']
                            dependent_pos = tree.nodes[edgex[1]]['upostag']
                            dep_intervener_rand = get.intervener_pos(edgex)
                            dep_relation_rand = get.relation_pos(edgex)
                            dep_head_rand = get.head_pos(edgex)

                            # Accumulate intervener POS tags for random baseline
                            for pos_tag in dep_intervener_rand:
                                if pos_tag in intervener_pos_tags_rand:
                                    intervener_pos_tags_rand[pos_tag] += 1
                                else:
                                    intervener_pos_tags_rand[pos_tag] = 1

                            projectivity_rand = 1 if find.is_projective(edgex) else 0
                            edge_degree_rand = find.edge_degree(edgex)  # gives the no. of edges crossing an edge
                            endpoint_cross_rand = find.endpoint_crossing(edgex)  # no. of heads which immediately dominates the nodes which causes non-projectivity in an edge span
                            HDD_rand = find.hdd(edgex)

                            with open('DL_matched_RLAs.csv', 'a') as results1:
                               results1.write(
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                        lang, "random", sent_id, n, projection_degree_rand, edgex, direction_rand,
                                        dep_distance_rand, dep_depth_rand, head_pos, dependent_pos, dep_relation_rand,
                                        dep_intervener_rand, dep_head_rand, max_arity_rand, avg_arity_rand
                                    ))

                    # Computes the measures for the real tree
                    max_arity_real = get.arity()[0]  # gives maximum arity present in the tree
                    avg_arity_real = get.arity()[1]
                    projection_degree_real = get.projection_degree(0)  # gives the projection degree of the tree i.e., size of longest projection chain in the tree
                    gap_degree_real = get.gap_degree(0)  # gives gap_degree of the tree
                    k_illnest_real = get.illnestedness(0, gap_degree_real)

                    for edgey in tree.edges:
                        if edgey[0] != 0 and edgey[0] in tree.nodes and edgey[1] in tree.nodes:
                            direction_real = get.dependency_direction(edgey)  # direction of the edge in terms of relative linear order of head and its dependent
                            dep_distance_real = get.dependency_distance(edgey)  # gives the distance between nodes connected by an edge
                            dep_depth_real = get.dependency_depth(edgey)
                            head_pos = tree.nodes[edgey[0]]['upostag']
                            dependent_pos = tree.nodes[edgey[1]]['upostag']
                            dependency_relation = tree[edgey[0]][edgey[1]]['drel']
                            dep_intervener_real = get.intervener_pos(edgey)
                            dep_relation_real = get.relation_pos(edgey)
                            dep_head_real = get.head_pos(edgey)

                            # Accumulate intervener POS tags for real baseline
                            for pos_tag in dep_intervener_real:
                                if pos_tag in intervener_pos_tags_real:
                                    intervener_pos_tags_real[pos_tag] += 1
                                else:
                                    intervener_pos_tags_real[pos_tag] = 1

                            projectivity_real = 1 if get.is_projective(edgey) else 0
                            edge_degree_real = get.edge_degree(edgey)  # gives the no. of edges crossing an edge
                            endpoint_cross_real = get.endpoint_crossing(edgey)  # no. of heads which immediately dominates the nodes which causes non-projectivity in an edge span
                            HDD_real = get.hdd(edgey)

                            with open('DL_matched_RLAs.csv', 'a') as results2:
                                results2.write(
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                        lang, "real", sent_id, n, projection_degree_real, edgey, direction_real,
                                        dep_distance_real, dep_depth_real, head_pos, dependent_pos, dep_relation_real,
                                        dep_intervener_real, dep_head_real, max_arity_real, avg_arity_real
                                    ))

                    print("\n-----------------\n" + str(tree.edges))







def get_sorted_intervener_tags(intervener_pos_tags):
    sorted_tags = {k: v for k, v in sorted(intervener_pos_tags.items(), key=lambda item: item[1], reverse=True)}
    return sorted_tags

# Assuming you have already accumulated intervener POS tags dictionaries for both real and random baselines:
# intervener_pos_tags_real and intervener_pos_tags_rand

# Generate and save the histogram of intervener POS tags for both random and real baselines
plt.figure(figsize=(12, 6))

# Real Baseline
plt.subplot(1, 2, 1)
real_intervener_tags = get_sorted_intervener_tags(intervener_pos_tags_real)
plt.bar(real_intervener_tags.keys(), real_intervener_tags.values())
plt.xlabel('Intervener POS Tags (Real Baseline)')
plt.ylabel('Frequency')
plt.title('Distribution of Intervener POS Tags (Real Baseline)')
plt.xticks(rotation=90)
plt.tight_layout()

# Random Baseline
plt.subplot(1, 2, 2)
rand_intervener_tags = get_sorted_intervener_tags(intervener_pos_tags_rand)
plt.bar(rand_intervener_tags.keys(), rand_intervener_tags.values())
plt.xlabel('Intervener POS Tags (Random Baseline)')
plt.ylabel('Frequency')
plt.title('Distribution of Intervener POS Tags (Random Baseline)')
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig('intervener_pos_tags_histogram_real_vs_random.png')
plt.show()


# Aggregate measures for real and random baselines
real_measures = [max_arity_real, avg_arity_real, projection_degree_real, gap_degree_real, k_illnest_real]
random_measures = [max_arity_rand, avg_arity_rand, projection_degree_rand, gap_degree_rand, k_illnest_rand]
labels = ['Max Arity', 'Avg Arity', 'Projection Degree', 'Gap Degree', 'Illnestedness']

# Pie chart for real baseline
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(real_measures, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Measures for Real Baseline')

# Pie chart for random baseline
plt.subplot(1, 2, 2)
plt.pie(random_measures, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Measures for Random Baseline')

plt.tight_layout()
plt.savefig('measures_comparison_pie_chart.png')
plt.show()

import seaborn as sns
import numpy as np

# Prepare data for heatmap
measures = np.array([
    real_measures,
    random_measures
])

# Labels for heatmap
row_labels = ['Real Baseline', 'Random Baseline']
column_labels = labels

plt.figure(figsize=(10, 6))
sns.heatmap(measures, annot=True, cmap='YlGnBu', fmt='.2f', xticklabels=column_labels, yticklabels=row_labels)
plt.title('Comparison of Measures: Real vs Random Baseline')
plt.xlabel('Measures')
plt.ylabel('Baseline Type')

plt.savefig('measures_comparison_heatmap.png')
plt.show()



# Bar charts for measures (example for Max Arity and Avg Arity)
plt.figure(figsize=(12, 6))

# Example: Max Arity comparison
plt.subplot(2, 2, 1)
plt.bar(['Real', 'Random'], [max_arity_real, max_arity_rand])
plt.title('Max Arity Comparison')
plt.ylabel('Max Arity')

# Example: Avg Arity comparison
plt.subplot(2, 2, 2)
plt.bar(['Real', 'Random'], [avg_arity_real, avg_arity_rand])
plt.title('Avg Arity Comparison')
plt.ylabel('Avg Arity')

plt.tight_layout()
plt.show()

# Network graph visualization of a random baseline tree (treex)
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(treex)  # or any other layout you prefer
nx.draw(treex, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
plt.title('Dependency Tree Visualization (Random Baseline)')
plt.show()

# Network graph visualization of a real baseline tree (tree)
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(tree)  # or any other layout you prefer
nx.draw(tree, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
plt.title('Dependency Tree Visualization (Real Baseline)')
plt.show()


# Intervener POS tag cloud for random baseline
intervener_pos_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(intervener_pos_tags_rand)
plt.figure(figsize=(10, 6))
plt.imshow(intervener_pos_cloud, interpolation='bilinear')
plt.title('Intervener POS Tag Cloud (Random Baseline)')
plt.axis('off')
plt.show()

# Intervener POS tag cloud for real baseline
intervener_pos_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(intervener_pos_tags_real)
plt.figure(figsize=(10, 6))
plt.imshow(intervener_pos_cloud, interpolation='bilinear')
plt.title('Intervener POS Tag Cloud (Real Baseline)')
plt.axis('off')
plt.show()
