import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os


class EchoNetDataset(Dataset):
    def __init__(self, root_dir, split="train", target_frames=32, transform=None):
        self.root_dir = root_dir
        self.target_frames = target_frames
        self.transform = transform
        df = pd.read_csv(os.path.join(root_dir, "FileList.csv"))
        self.files = df[df["Split"] == split].reset_index(drop=True)
        self.video_dir = os.path.join(root_dir, "Videos")
        self.tracings = pd.read_csv(os.path.join(root_dir, "VolumeTracings.csv"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files.iloc[idx]["FileName"] + ".avi"
        ef = self.files.iloc[idx]["EF"] / 100.0
        video_path = os.path.join(self.video_dir, filename)

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = np.array(frames)
        num_frames = len(frames)
        if num_frames < self.target_frames:
            padding = np.zeros((self.target_frames - num_frames, 112, 112, 3))
            frames = np.concatenate([frames, padding], axis=0)
        elif num_frames > self.target_frames:
            frames = frames[:self.target_frames]

        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0

        # Real segmentation masks
        mask = self._get_segmentation_mask(filename.split(".")[0], num_frames)
        mask = torch.from_numpy(mask).float()

        if self.transform:
            frames = self.transform(frames)

        return frames, ef, mask

    def _get_segmentation_mask(self, filename, num_frames):
        mask = np.zeros((self.target_frames, 1, 112, 112))
        file_tracings = self.tracings[self.tracings["FileName"] == filename]
        for frame in range(min(num_frames, self.target_frames)):
            frame_data = file_tracings[file_tracings["Frame"] == frame]
            if not frame_data.empty:
                points = np.array(list(zip(frame_data["X"], frame_data["Y"])), dtype=np.int32)
                cv2.fillPoly(mask[frame, 0], [points], 1)
        return mask


def collate_fn(batch):
    frames, efs, masks = zip(*batch)
    frames = torch.stack(frames)
    efs = torch.tensor(efs, dtype=torch.float32)
    masks = torch.stack(masks)
    return frames, efs, masks