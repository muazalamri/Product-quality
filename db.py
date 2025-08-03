import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from random import shuffle
def data():
    root='casting_data/casting_data/train/'
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128,128)),
        transforms.ConvertImageDtype(torch.float32)
    ])
    train_ok=os.listdir(root+'ok_front')
    train_def=os.listdir(root+'def_front')
    sample=min(len(train_ok),len(train_def))
    train_ok=train_ok[:sample]
    train_def=train_def[:sample]
    train_data=[
        {'data':transform(read_image(root+'ok_front/'+str(i))),'label':torch.tensor([0.999999])} for i in train_ok
    ]+[
        {'data':transform(read_image(root+'def_front/'+str(i))),'label':torch.tensor([0.0000001])} for i in train_def
    ]
    shuffle(train_data)
    shuffle(train_data)
    return train_data
