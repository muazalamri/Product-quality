import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from random import shuffle
from model import Model as model
model.load_state_dict(torch.load('model_0_0.9445410628019324.pt'))#,device='cpu'))
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
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for i in test_data:
        outputs = model(i['data'].unsqueeze(0))
        preds = (outputs > 0.5).float()  # لأن الخرج من Sigmoid
        y_pred.append(int(preds.cpu().numpy()))
        y_true.append(int(i['label']>0.1))
print('pred=',y_pred,'\ntrue=',y_true)
from sklearn.metrics import classification_report

report=str(classification_report(y_true, y_pred, target_names=["ok", "def"]))
print(report)
open('report.md','w',encoding='utf-8').write(report)
