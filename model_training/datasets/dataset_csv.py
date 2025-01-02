import torch
import pandas as pd
from torch.utils.data import Dataset

class IKDatasetCSV(Dataset):
    def __init__(self, file_path, with_orientation=True):
        self.data = pd.read_csv(file_path)
        self.with_orientation = with_orientation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read from CSV
        x = self.data.iloc[idx]['TCP_X']
        y = self.data.iloc[idx]['TCP_Y']
        z = self.data.iloc[idx]['TCP_Z']

        rx = self.data.iloc[idx]['Orientation_X']
        ry = self.data.iloc[idx]['Orientation_Y']
        rz = self.data.iloc[idx]['Orientation_Z']

        # The 6 joint angles
        j1 = self.data.iloc[idx]['Shoulder_Pan']
        j2 = self.data.iloc[idx]['Shoulder_Lift']
        j3 = self.data.iloc[idx]['Elbow']
        j4 = self.data.iloc[idx]['Wrist_1']
        j5 = self.data.iloc[idx]['Wrist_2']
        j6 = self.data.iloc[idx]['Wrist_3']

        if self.with_orientation:
            # Positions => x,y,z,rx,ry,rz
            positions = torch.tensor([x, y, z, rx, ry, rz], dtype=torch.float32)
            # Joint angles => 6 + 3 orientation => 9D
            joint_angles = torch.tensor([j1, j2, j3, j4, j5, j6,
                                         rx, ry, rz],
                                        dtype=torch.float32)
        else:
            # Positions => just x,y,z
            positions = torch.tensor([x, y, z], dtype=torch.float32)
            # Joint angles => 6 + 3 orientation => 9D
            joint_angles = torch.tensor([j1, j2, j3, j4, j5, j6,
                                         rx, ry, rz],
                                        dtype=torch.float32)
        return positions, joint_angles


class IKDatasetValCSV(Dataset):
    """
    Same as above, but for validation data (optional).
    """
    def __init__(self, file_path, with_orientation=True):
        self.data = pd.read_csv(file_path)
        self.with_orientation = with_orientation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This can mirror the same logic as IKDatasetCSV
        x = self.data.iloc[idx]['TCP_X']
        y = self.data.iloc[idx]['TCP_Y']
        z = self.data.iloc[idx]['TCP_Z']

        rx = self.data.iloc[idx]['Orientation_X']
        ry = self.data.iloc[idx]['Orientation_Y']
        rz = self.data.iloc[idx]['Orientation_Z']

        j1 = self.data.iloc[idx]['Shoulder_Pan']
        j2 = self.data.iloc[idx]['Shoulder_Lift']
        j3 = self.data.iloc[idx]['Elbow']
        j4 = self.data.iloc[idx]['Wrist_1']
        j5 = self.data.iloc[idx]['Wrist_2']
        j6 = self.data.iloc[idx]['Wrist_3']

        if self.with_orientation:
            positions = torch.tensor([x, y, z, rx, ry, rz], dtype=torch.float32)
            joint_angles = torch.tensor([j1, j2, j3, j4, j5, j6,
                                         rx, ry, rz],
                                        dtype=torch.float32)
        else:
            positions = torch.tensor([x, y, z], dtype=torch.float32)
            joint_angles = torch.tensor([j1, j2, j3, j4, j5, j6,
                                         rx, ry, rz],
                                        dtype=torch.float32)
        return positions, joint_angles
