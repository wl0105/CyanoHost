import torch
import os
import sys
from torch_geometric.data import Data
from Bio import SeqIO
import multiprocessing


prokka_path = ""
esm_path = ""

def graph_feature(seq_id):
    d_seqs = {}
    features = []
    faa_file = "%s/%s/%s.faa" % (prokka_path, seq_id.strip(),seq_id.strip())
    
    for seq_record in SeqIO.parse(faa_file, "fasta"):
        d_seqs[seq_record.id] = str(seq_record.seq)
    
    edge_index = set()

    for k in d_seqs.keys():
        esm_tensor_path = esm_path + '/' + k.strip() + '.pt'
        esm_tensor = torch.load(esm_tensor_path, map_location=torch.device('cpu'))
        esm_tensor = esm_tensor['mean_representations'][36]

        features.append(esm_tensor)

        for i in range(len(d_seqs.keys())):
            edge_index.add((i, (i - 1) % len(d_seqs.keys())))
            edge_index.add((i, (i - 2) % len(d_seqs.keys())))
            edge_index.add((i, (i + 1) % len(d_seqs.keys())))
            edge_index.add((i, (i + 2) % len(d_seqs.keys())))

    edge_index = torch.tensor(list(edge_index), dtype=torch.long).t()
    return features, edge_index


def process_sequence(seq_id, train_save_path):
    torch.cuda.empty_cache()
    cds_feature, cds_edge_index = graph_feature(seq_id)
    
    cds_feature = [tensor.detach() for tensor in cds_feature]
    cds_feature = torch.stack(cds_feature)
    cds_feature = cds_feature.clone().detach().type(torch.float)
    
    data = Data(x=cds_feature, edge_index=cds_edge_index)
    save_path = train_save_path + '/' + seq_id + '.pt'
    torch.save(data, save_path)


def new_dataset(label_path, save_path):
    sequences=[]

    label_f = open(label_path, 'r')
    for line in label_f:
        sequences.append(line.split('\t')[0])
    label_f.close()

    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(process_sequence, [(seq_id, save_path) for seq_id in sequences])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_label_path = sys.argv[1]
    train_save_path = sys.argv[2]
    prokka_path = sys.argv[3]  
    esm_path = sys.argv[4]     

    os.makedirs(train_save_path)
    new_dataset(train_label_path, train_save_path)





