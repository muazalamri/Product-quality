import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from random import shuffle
from model import Model as model
from time import time
model.load_state_dict(torch.load('models/model_0_0.9490086956521739.pt'))#,device='cpu'))
root='casting_data/casting_data/test/'
transform = transforms.Compose([
  transforms.Grayscale(1),
  transforms.Resize((128,128)),
  transforms.ConvertImageDtype(torch.float32)
])
test_ok=os.listdir(root+'ok_front')
test_def=os.listdir(root+'def_front')
sample=min(len(test_ok),len(test_def))
test_ok=test_ok[:sample]
tsst_def=test_def[:sample]

test_data=[
  {'data':transform(read_image(root+'ok_front/'+str(i))),'label':torch.tensor([0.999999])} for i in test_ok
 ]+[
  {'data':transform(read_image(root+'def_front/'+str(i))),'label':torch.tensor([0.0000001])} for i in test_def
]
print(len(test_data),len(test_def),len(test_ok))
print(sample*2)
model.eval()

now=time()
with torch.no_grad():
    for i in test_data:
        outputs = model(i['data'].unsqueeze(0))
print((time()-now)/(len(test_data)))
