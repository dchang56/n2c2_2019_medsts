"""

This notebook just processes UMLS META files and creates snomed_graph

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
import networkx as nx

META_dir = '/home/dc925/umls/2019AA-full/2019AA/META/'
concepts_file = os.path.join(META_dir, 'MRCONSO.RRF')
semtypes_file = os.path.join(META_dir, 'MRSTY.RRF')
relations_file = os.path.join(META_dir, 'MRREL.RRF')

#semtypes_df maps Concepts in SNOMEDCT_US to their semantic types (some have more than 1 semtypes)
semtypes_df = pd.read_csv(semtypes_file, sep='|', header=None)
semtypes_df = semtypes_df[[0, 3]]
semtypes_df.columns = ['CUI', 'STY']

##Concepts
concepts_df = pd.read_csv(concepts_file, sep='|', header=None)
concepts_df = concepts_df[[0, 2, 11, 14]]
concepts_df.columns = ['CUI', 'TS', 'SAB', 'STR'] 
concepts_df = concepts_df[concepts_df['SAB']=='SNOMEDCT_US'] #only keep snomedct
concepts_df = concepts_df[concepts_df['TS']=='P'] #filter CUIs to only Preferred terms
concepts_df = concepts_df.drop_duplicates(subset=['CUI','TS']) #drop duplicates by taking the first preferred string
concepts_df = concepts_df[['CUI', 'STR']] #discard TS and SAB
concepts_df.shape

##Relations
relations_df = pd.read_csv(relations_file, sep='|', header=None)
relations_df = relations_df[[0, 3, 4, 7, 10]]
relations_df.columns = ['CUI1', 'REL', 'CUI2', 'RELA', 'SAB']
relations_df = relations_df[relations_df['SAB']=='SNOMEDCT_US'] #filter for snomedct
relations_df = relations_df[['CUI1','REL','CUI2','RELA']]
relations_df.index = relations_df['RELA'].str.len() #sort by len of RELA so we can get more info about relation
relations_df = relations_df.sort_index(ascending=False).reset_index(drop=True)
relations_df = relations_df.drop_duplicates(subset=['CUI1','CUI2'], keep='first') #drop duplicate relations
relations_df = relations_df[relations_df['CUI1']!=relations_df['CUI2']] #49138 self-connections; remove; to be added in dgl

#Relation summary: 6 relation types, 236 specific (symmetric)
#4M rels for 390k concepts
#We'll end up w 2M relations since undirected graph will eliminate the inverse relations

## only keep cuis that have relations in snomedct graph
cuis = pd.unique(pd.concat([relations_df['CUI1'],relations_df['CUI2']], axis=0))
concepts_df = concepts_df[concepts_df['CUI'].isin(cuis)]

#convert Dataframe to dicts, and create concepts to ids and strings mappings
concepts_dicts = concepts_df.to_dict(orient='records')
i = 0
concepts_to_ids = {}
concepts_to_strings = {}
for concept in concepts_dicts:
    concepts_to_strings[concept['CUI']] = concept['STR']
    concepts_to_ids[concept['CUI']] = i
    i += 1
relation_dicts = relations_df.to_dict(orient='records')

graph = nx.Graph()
for relation in relation_dicts:
    cui1, cui2 = relation['CUI1'], relation['CUI2']
    graph.add_node(concepts_to_ids[cui1])
    graph.add_node(concepts_to_ids[cui2])
    graph.add_edge(concepts_to_ids[cui1], concepts_to_ids[cui2])

print(graph.number_of_nodes())
print(graph.number_of_edges())
#390k nodes with 2M edges
#Only 72MB!

nx.write_gpickle(graph, "SNOMEDCT_Graph.pkl")

concepts_to_ids_file = 'concepts_to_ids.json'
concepts_to_strings_file = 'concepts_to_strings.json'
semtypes_file = 'semtypes.csv'
np.save(concepts_to_ids_file, concepts_to_ids)
np.save(concepts_to_strings_file, concepts_to_strings)
semtypes_df.to_csv(semtypes_file,index=False)













