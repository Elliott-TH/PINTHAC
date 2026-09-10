import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.3'
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_optimizer.optimizer.soap import SOAP as Soap
from scipy.optimize import fsolve 
import SCW_Props as scw
import IAPWS_95 as iapws
from Broyden import SSBroyden
torch.manual_seed(3472)
import os
#os.environ['HIP_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(3472)
print(f"Using device: {device}")


w1=128#
n=1
nx = 100
print('here')
table1 = '/home/elliott/Draft_Props/TPrho_table.txt'
table2 = '/home/elliott/Draft_Props/Table_2.txt'
#table3 = '/home/elliott/Draft_Props/Table_3.txt'
table3 = '/home/elliott/Density_Data_95.txt'
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.Inputx = nn.Linear(1,w1)
        self.Layer1x = nn.Linear(w1,w1)
        self.Layer2x = nn.Linear(w1,w1)
        self.Layer3x = nn.Linear(w1,w1)
        self.Layer4x = nn.Linear(w1,w1)
        self.Outputx = nn.Linear(w1,w1)

        self.Inputy = nn.Linear(nx,w1)
        self.Layer1y = nn.Linear(w1,w1)
        self.Layer2y = nn.Linear(w1,w1)
        self.Layer3y = nn.Linear(w1,w1)
        self.Layer4y = nn.Linear(w1,w1)
        self.Outputy = nn.Linear(w1,w1)
        self.Tanh = nn.Tanh()
        #self.Relu = nn.ReLU()
        #self.k = nn.Parameter(torch.tensor([0.01]))
        self.Softplus = nn.Softplus()
        self.Softmax = nn.Softmax()
        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        #self.Sin = torch.sin()
        self.gelu = nn.GELU()
        #self.Sin = torch.sin()

    def act(self,x):
        return self.gelu(x)#self.Tanh(x)
    
    def forward(self,x,f):
        x1 = self.Inputx(x)
        x1 = PINN.act(self,x1)#PINN.act(x1)
        x1 = self.Layer1x(x1)
        x1 = PINN.act(self,x1)
        x1 = self.Layer2x(x1)
        x1 = PINN.act(self,x1)
        x1 = self.Layer3x(x1)
        x1 = PINN.act(self,x1)
        x1 = self.Layer4x(x1)
        x1 = PINN.act(self,x1)
        xout = self.Outputx(x1)

        y1 = self.Inputy(f)
        y1 = PINN.act(self,y1)#PINN.act(x1)
        y1 = self.Layer1y(y1)
        y1 = PINN.act(self,y1)
        y1 = self.Layer2y(y1)
        y1 = PINN.act(self,y1)
        y1 = self.Layer3y(y1)
        y1 = PINN.act(self,y1)
        y1 = self.Layer4y(y1)
        y1 = PINN.act(self,y1)
        yout = self.Outputy(y1)
        
        net_out = torch.sum(xout * yout, dim=1, keepdim=True)

        return net_out


model = PINN().to(device)



x = torch.linspace(0,1,nx,device=device)


Nmax=15
def rand(x,evodd='None'):
    N= torch.randint(1,Nmax+1,(1,1),device=device).item()
    coeff = 2*torch.rand(N,device=device)-1
    coeff2 = 2*torch.rand(N,device=device)-1
    n = torch.arange(0,N,device=device)
    n.reshape(1,N)
    if evodd == 'None':
        n = n
    elif evodd == 'even':
        n = 2*n
    elif evodd == 'odd':
        n = 2*n+1
    x = x.unsqueeze(dim=1)
    xN = x**n #/ torch.lgamma(n+1).exp()
    f_n = xN * coeff
    f_out = f_n.sum(dim=1)
    f_x = (n[1:]*x**(n[1:]-1) * coeff[1:]).sum(dim=1)

    f_out2 = coeff*torch.sin(np.pi*n*x)+coeff2*torch.cos(np.pi*n*x)
    f_x2 = np.pi*n*(coeff*torch.cos(np.pi*n*x)-coeff2*torch.sin(np.pi*n*x))

    f_out2 = f_out2.sum(dim=1)
    f_x2 = f_x2.sum(dim=1)
    #fout = torch.cat([])
    return f_out2, f_x2

def GenTrain(x,nfunc):
    y=torch.tensor((),device=device)
    dy = torch.tensor((),device=device)
    for i in range(nfunc):
        y_i,dy_i = rand(x)
        y = torch.concat([y,y_i],dim=0)
        dy = torch.concat([dy,dy_i],dim=0)
    x = x.repeat(nfunc)
    Xtrain = torch.concat([x.unsqueeze(dim=1),y.unsqueeze(dim=1)],dim=1)
    F_out = dy.unsqueeze(dim=1)
    return Xtrain,F_out



Nfunc=100

X, F = GenTrain(x,Nfunc)
print(X)
xtr = X[:,0:1]#.unsqueeze(dim=1)
ftr = X[:,1:2].reshape(Nfunc,nx)
print(ftr)
ftr = torch.repeat_interleave(ftr,nx,dim=0)
print(ftr)

f_x = F.reshape(Nfunc,nx)
f_x = torch.repeat_interleave(f_x,nx,dim=0)
print(f_x.size())
print(xtr.size())

def Loss(model, xin, fin):
    F_pred = model(xin, fin)
    l2 = F_pred - F
    loss = torch.mean(l2**2)
    return loss

epochs =500
loss_history = []
print("training")


#optimizer = SSBroyden(model.parameters(), lr=1.0, memory_size=20)

optimizer = Soap(model.parameters())
#optimizer=optim.Adam(model.parameters(),lr=0.001)
loss_history = []
testloss = []

from tqdm import tqdm

pbar = tqdm(range(epochs + 1), desc="Training")
for epoch in pbar:
    optimizer.zero_grad()
    output = model(xtr,ftr)
    loss = Loss(model, xtr,ftr)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    #with torch.no_grad():
    #    test_loss = torch.mean((model(Xtest)-rhotest)**2)
    #    testloss.append(test_loss)

    pbar.set_postfix(loss=f"{loss.item():.4e}")

    #if epoch % 10 == 0:
    #    print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}")



epochs =100

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn="strong_wolfe"
)

loss_history = []

for epoch in range(epochs + 1):

    def closure():
        optimizer.zero_grad()

        output = model(xtr,ftr)   # keep if Loss depends on forward pass
        loss = Loss(model, xtr,ftr)

        loss.backward()

        return loss

    loss = optimizer.step(closure)

    loss_history.append(loss.item())

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

    if loss.item() <= 5e-6:
        break


x = torch.linspace(0,1,nx,device=device)
y = torch.sin(np.pi*x)
Xplt = torch.concat([x.unsqueeze(dim=1),y.unsqueeze(dim=1)],dim=1)

with torch.no_grad():
    y_x = model(x.unsqueeze(dim=1),y)
    y_x = y_x.detach().cpu().numpy()

xplt = np.linspace(0,1,nx)
y_x_true = np.pi*np.cos(np.pi*xplt)

plt.plot(xplt,y_x,label='Pred')
plt.plot(xplt,y_x_true,label='True')
plt.legend()
plt.show()