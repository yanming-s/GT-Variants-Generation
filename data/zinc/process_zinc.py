import os
import os.path as osp
import pathlib
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_tar
from torch_geometric.data.lightning import LightningDataset


ATOM_LIST = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
BOND_LIST = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
RAW_URL = "http://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc15_250K_2D.tar.gz"


class ZincDatasetModule(InMemoryDataset):
    def __init__(self, split, num, root, transform = None, pre_transform = None, pre_filter = None):
        self.split = split
        if split == "train":
            self.split_idx = 0
        elif split == "val":
            self.split_idx = 1
        else:
            self.split_idx = 2
        self.num = num
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.split_idx], weights_only=False)
    
    @property
    def raw_file_names(self):
        # Expected raw files in raw_dir
        return ["zinc15_250K_2D.csv"]

    @property
    def split_file_name(self):
        # name for each data file
        return ["train_zinc.csv", "val_zinc.csv", "test_zinc.csv"]

    @property
    def split_paths(self):
        return [osp.join(self.raw_dir, f) for f in self.split_file_name]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        try:
            file_path = download_url(RAW_URL, self.raw_dir)
            extract_tar(file_path, self.raw_dir, mode='r:gz')
            os.unlink(file_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading ZINC dataset: {e}")
        dataset = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))
        # split the dataset
        n_train = self.num["train"]
        n_val = self.num["val"]
        n_test = self.num["test"]
        train, val, test, _ = np.split(
            dataset.sample(frac=1, random_state=0),
            [n_train, n_val + n_train, n_val + n_train + n_test]
        )
        # save the split dataset
        train.to_csv(os.path.join(self.raw_dir, 'train_zinc.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val_zinc.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test_zinc.csv'))

    def process(self):
        # atom and bond types
        atom_types = {atom: i for i, atom in enumerate(ATOM_LIST)}
        bond_types = {bond: i for i, bond in enumerate(BOND_LIST)}
        # process the data
        raw_path = self.split_paths[self.split_idx]
        dataset = pd.read_csv(raw_path)
        smile_list = dataset["smiles"].tolist()
        data_list = []
        for _, smile in enumerate(tqdm(smile_list, desc=f"Processing {self.split} data")):
            mol = Chem.MolFromSmiles(smile)
            num = mol.GetNumAtoms()
            # atom processing
            type_idx = []
            for atom in mol.GetAtoms():
                atom: Chem.Atom
                type_idx.append(atom_types[atom.GetSymbol()])
            # bond processing
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                bond: Chem.Bond
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += [bond_types[bond.GetBondType()]] * 2
            if len(row) == 0:
                continue
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(BOND_LIST)+1).float()
            perm = (edge_index[0] * num + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            x = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_LIST)).float()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
        # save the processed data
        torch.save(self.collate(data_list), self.processed_paths[self.split_idx])
        smiles_save_path = osp.join(self.processed_dir, f"{self.split}_smiles.csv")
        dataset["smiles"].to_csv(smiles_save_path, index=False)


class ZincDataset(LightningDataset):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = osp.join(base_path, self.datadir)
        nums = {
            "train": cfg.dataset.train_size,
            "val": cfg.dataset.val_size,
            "test": cfg.dataset.test_size
        }
        train_dataset = ZincDatasetModule("train", nums, root_path)
        val_dataset = ZincDatasetModule("val", nums, root_path)
        test_dataset = ZincDatasetModule("test", nums, root_path)
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=cfg.train.batch_size
        )
    
    def __getitem__(self, idx):
        return self.train_dataset[idx]
    
    def get_train_smiles(self):
        return pd.read_csv(osp.join(self.train_dataset.processed_dir, "train_smiles.csv"))["smiles"].tolist()
