import os
import torch
import pandas as pd
from torch_geometric.data import Data

def get_raw_text_arxiv_2023(use_text=True, base_path="dataset/arxiv_2023"):
    # Load processed data
    edge_index = torch.load(os.path.join(base_path, "processed", "edge_index.pt"))
    
    # Load raw data
    # edge_df = pd.read_csv(os.path.join(base_path, "raw", "edge.csv.gz"), compression='gzip')
    titles_df = pd.read_csv(os.path.join(base_path, "raw", "titles.csv.gz"), compression='gzip')
    abstracts_df = pd.read_csv(os.path.join(base_path, "raw", "abstracts.csv.gz"), compression='gzip')
    ids_df = pd.read_csv(os.path.join(base_path, "raw", "ids.csv.gz"), compression='gzip')
    labels_df = pd.read_csv(os.path.join(base_path, "raw", "labels.csv.gz"), compression='gzip')
    
    # Load split data
    train_id_df = pd.read_csv(os.path.join(base_path, "split", "train.csv.gz"), compression='gzip')
    val_id_df = pd.read_csv(os.path.join(base_path, "split", "valid.csv.gz"), compression='gzip')
    test_id_df = pd.read_csv(os.path.join(base_path, "split", "test.csv.gz"), compression='gzip')
    
    num_nodes = len(ids_df)
    titles = titles_df['titles'].tolist()
    abstracts = abstracts_df['abstracts'].tolist()
    ids = ids_df['ids'].tolist()
    labels = labels_df['labels'].tolist()
    train_id = train_id_df['train_id'].tolist()
    val_id = val_id_df['val_id'].tolist()
    test_id = test_id_df['test_id'].tolist()

    features = torch.load(os.path.join(base_path, "processed", "features.pt"))

    y = torch.load(os.path.join(base_path, "processed", "labels.pt"))
    
    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])
    
    data = Data(
        x=features,
        y=y,
        paper_id=ids,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )
    
    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id
    
    if not use_text:
        return data, None
    
    text = {'title': titles, 'abs': abstracts, 'label': labels, 'id': ids}
    
    return data, text
