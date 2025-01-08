import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

# TODO include orientation

class IKDatasetCSV(Dataset):
    def __init__(self, file_path, with_orientation=False):
        self.data = pd.read_csv(file_path)
        self.with_orientation = with_orientation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract positions (TCP_X, TCP_Y, TCP_Z)
        positions = torch.Tensor([
            self.data.iloc[idx]['TCP_X'],
            self.data.iloc[idx]['TCP_Y'],
            self.data.iloc[idx]['TCP_Z']
        ])
        
        # Extract joint angles (Shoulder_Pan, Shoulder_Lift, Elbow, Wrist_1, Wrist_2, Wrist_3)
        joint_angles = torch.Tensor([
            self.data.iloc[idx]['Shoulder_Pan'],
            self.data.iloc[idx]['Shoulder_Lift'],
            self.data.iloc[idx]['Elbow'],
            self.data.iloc[idx]['Wrist_1'],
            self.data.iloc[idx]['Wrist_2'],
            self.data.iloc[idx]['Wrist_3']
        ])

        return positions, joint_angles
    


class IKDatasetValCSV(Dataset):
    def __init__(self, file_path, with_orientation=False):
        self.data = pd.read_csv(file_path)
        self.with_orientation = with_orientation

    def __len__(self):
        return 400

    def __getitem__(self, idx):
        reversed_idx = len(self.data) - idx - 1

        positions = torch.Tensor([
            self.data.iloc[reversed_idx]['TCP_X'],
            self.data.iloc[reversed_idx]['TCP_Y'],
            self.data.iloc[reversed_idx]['TCP_Z']
        ])

        joint_angles = torch.Tensor([
            self.data.iloc[reversed_idx]['Shoulder_Pan'],
            self.data.iloc[reversed_idx]['Shoulder_Lift'],
            self.data.iloc[reversed_idx]['Elbow'],
            self.data.iloc[reversed_idx]['Wrist_1'],
            self.data.iloc[reversed_idx]['Wrist_2'],
            self.data.iloc[reversed_idx]['Wrist_3']
        ])

        return positions, joint_angles
