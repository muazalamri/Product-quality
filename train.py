import torch
from db import data as data_fun
from model import Model as model
device='cpu'
optim=torch.optim.Adam(lr=1e-4,params=model.parameters())
loss_fun=torch.nn.BCELoss()
index=0
count=0
points=0
for i in range(10):
    for i in data_fun():
        data = i['data'].to(device).unsqueeze(0)  # [1, C, H, W]
        label = i['label'].to(device)
        output = model(data)                      # [1, num_classes]
        loss = loss_fun(output, label.unsqueeze(0))  # label: [1]
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (output<0.5 and label<0.3)|(output>=0.5 and label>0.7):points+=1
        count+=1
        ac=points/count
    print(loss.item(),' ,for class:',label,',ac:',ac)
    torch.save(model.state_dict(),f'model_{index}_{ac}.pt')
    print(f'saved:model_{index}_{ac}.pt')
    if index>1:
        points,count=0,0

