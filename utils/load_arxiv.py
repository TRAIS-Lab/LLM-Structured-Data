"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_arxiv.py
"""

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd


def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # data.edge_index = data.adj_t.to_symmetric()
    data.edge_index
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
    
    raw_text = pd.read_csv('dataset/ogbn_arxiv/titleabs.tsv', sep='\t')
    raw_text.columns = ['paper id', 'title', 'abs']

    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = {'title': [], 'abs': [], 'label': []}

    for ti, ab in zip(df['title'], df['abs']):
        text['title'].append(ti)
        text['abs'].append(ab)
    
    # Load the label index to arXiv category mapping data
    label_mapping_data = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz')
    label_mapping_data.columns = ['label_idx', 'arxiv_category']

    for i in range(len(data.y)):
        row = label_mapping_data.loc[label_mapping_data['label_idx'].isin(data.y[i].numpy())]
        # If the row doesn't exist, return a message indicating this
        if len(row) == 0:
            raise 'No matching arXiv category found for this label index.'
    
        # Parse the arXiv category string to be in the desired format 'cs.XX'
        arxiv_category = 'cs.' + row['arxiv_category'].values[0].split()[-1].upper()
        text['label'].append(arxiv_category)

    return data, text


def generate_arxiv_keys_list():
    label_mapping_data = pd.read_csv('dataset/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz', compression='gzip')
    label_mapping_data.columns = ['label_idx', 'arxiv_category']
    arxiv_categories = label_mapping_data['arxiv_category'].unique()
    return ['cs.' + category.split()[-1].upper() for category in arxiv_categories]
