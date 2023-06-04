import torch 
import decord as de
import torch.utils.data as data
ctx = de.cpu(0)
de.bridge.set_bridge("torch")

#@title UVG-HD
class VideoDataset(data.Dataset):
    def __init__(
        self, 
        video_path,
        frame_gap=1,
    ):
        self.frame_gap = frame_gap
        self.video = de.VideoReader(video_path, ctx=ctx)
    
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        idx = idx * self.frame_gap
        frame = self.video.get_batch([idx]).permute(0, 3, 1, 2)
        return torch.Tensor(idx), frame