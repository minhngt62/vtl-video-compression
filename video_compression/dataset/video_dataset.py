import torch 
import decord as de
import torch.utils.data as data
ctx = de.cpu(0)
de.bridge.set_bridge("torch")

import skvideo.io
import skvideo.datasets

# Use directly for video files
class VideoDataset(data.Dataset):
    def __init__(
        self, 
        video_path,
        frame_gap=1,
    ):
        self.frame_gap = frame_gap
        self.video = de.VideoReader(video_path, ctx=ctx)
    
    def __len__(self):
        return len(self.video) // self.frame_gap
    
    def __getitem__(self, idx):
        idx = idx * self.frame_gap
        frame = self.video.get_batch([idx]).permute(0, 3, 1, 2)
        idx = idx / (len(self.video) / self.frame_gap) # normalize idx
        return torch.tensor(idx), frame

# Use for bigbuckbunny loaded from skvideo
class BigBuckBunny(data.Dataset):
    def __init__(
        self,
        frame_gap=1
    ):
        self.frame_gap = frame_gap
        self.video = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
    
    def __len__(self):
        return self.video.shape[0] // self.frame_gap
    
    def __getitem__(self, idx):
        idx = idx * self.frame_gap
        frame = self.video[idx] / 255
        idx = idx / (len(self.video) / self.frame_gap) # normalize idx
        return torch.tensor(idx), torch.Tensor(frame).float().permute(2, 0, 1)
