from glob import glob
import os
import csv

def get_content(path, label):
    fic_list = []
    for file in glob(os.path.join(path, "*.txt")):
        lines = open(file).read(). splitlines()
        content_file = " ".join([line.strip() for line in lines]) 
        fic_list.append([content_file, label])
    return fic_list

path_pos = "projets_M1/corpus/imdb/pos/"
path_neg = "projets_M1/corpus/imdb/neg/"
fic=[]
fic.append(["comment", "label"])
fic += get_content(path_pos, "pos")
fic += get_content(path_neg, "neg")

with open('avis_imdb.tsv', 'wt') as csv_file:
    tsv_writer = csv.writer(csv_file, delimiter='\t')
    for file in fic:
        tsv_writer.writerow(file)