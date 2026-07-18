import torch
from torch.utils.data import TensorDataset, DataLoader
from models import JEPA
from train import *

BATCH_SIZE = 32
STATES_PER_SAMPLE = 5
NUM_SAMPLES = 1024
STATE_DIM = 73
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

states = torch.rand(
    NUM_SAMPLES,
    STATES_PER_SAMPLE,
    STATE_DIM,
)

loader = DataLoader(
    TensorDataset(states),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
jepa = JEPA(
    latent_dim=128,
    encoder_blocks=2,
    encoder_hdim=256,
    encoder_attheads=4,
    proj_blocks=2,
    proj_hdim=128,
    proj_attheads=4,
    momentum=0.995,
    obj_lengths=(12, 22, 16, 23),
    emb_hdim=256
)


optimizer = torch.optim.AdamW(
    jepa.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

train_mask(jepa, 100, loader, optimizer)
