import torch
import numpy as np
import random
import math
import sys
import argparse
import matplotlib.pyplot as plt
from torch.utils import tensorboard
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__) 

class FunctionGenerator:
    def __init__(self, n_inputs, features=None, default_value=None, func = lambda X: np.float32(np.sum(np.array(X),axis=-1))):
        if features is None:
            features = np.arange(n_inputs)
            
        if (n_inputs < len(features)):
            logger.error(f'n_inputs {n_inputs} is less than the length of features {len(features)}')

        self.n_inputs = n_inputs
        self.features = features
        self.default_value = default_value
        
        weights = np.zeros(n_inputs)
        for f in self.features:
            weights[f] = 1.0

        def func_wrapper(X):
            xd = np.array(X) * np.array(weights)
            logger.info( f'func_wrapper {X}*{weights}={xd} {xd.shape}')            
            for i,p in enumerate(weights):
                if (self.default_value is not None) and (i not in self.features):
                    xd[:,i] = default_value
            return func(xd)
        
        self.func = func_wrapper
                    
                                    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 g, 
                 n_samples=10, 
                 noise=0,
                 device = None):
        if (device is None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        #self.X = (torch.rand(samples, x_inputs + i_inputs + z_inputs, dtype=torch.float32) - 0.5)*2.0
        self.X = (torch.rand(n_samples, g.n_inputs, dtype=torch.float32))
        
        self.g = g
        #self.fun = np.vectorize(f, signature='(n,m)->(n)')
        yv = torch.tensor(self.g.func(self.X))
        
        self.Y = yv
        
        self.X = self.X.to(device=device)
        self.Y = self.Y.to(device=device)
        
        self.len = self.X.shape[0]
        self.shape = self.X.shape
        
    def loader(self, batch_size=256):
        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=True)
       
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
   
    def __len__(self):
        return self.len
    
    
class NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lossfn=None):
        super(NN, self).__init__()
        
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        #torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        
        if lossfn is None:
            lossfn = torch.nn.MSELoss()
        
        self.lossfn = lossfn
        self.epoch_count = 0
           
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x)).to(args.device)
        x = torch.nn.functional.relu(self.layer_2(x)).to(args.device)
        x = torch.nn.functional.relu(self.layer_3(x)).to(args.device)

        #x = torch.nn.functional.relu(self.layer_3(x)).to(args.device)

        return x
    
    
    def train(self, data, num_epochs=1, batch_size = 256, learning_rate=0.1, writer=None ):
        #optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        loss_values = []
        loss = 0
        for epoch in range(num_epochs):
            for batch, (X, Y) in enumerate(data.loader(batch_size=batch_size)):
                optimizer.zero_grad()
                yd = self(X)
                yl = Y.unsqueeze(-1)
                    
                #print(yd - Y.unsqueeze(-1))
                loss = self.lossfn(yd,Y.unsqueeze(-1))
                loss_values.append(loss.item())

                loss.backward()
                optimizer.step()
            if writer:
                writer.add_scalar("Loss/train", loss, self.epoch_count)
                self.epoch_count = self.epoch_count + 1
            # if (epoch % 500 == 0):
            #     print( f'epoch {epoch+1} loss={loss}')   
        if writer:
            writer.flush()      
        return loss_values
    
    def test(self, data):
        loader = data.loader()
        sum_vloss = []
        with torch.no_grad():
            for i, (x,y) in enumerate(loader):
                yd = self(x)
                vloss = (yd - y.unsqueeze(-1))**2
                sum_vloss.extend(vloss.flatten().tolist())
        return sum_vloss

def draw_truth(ax1, g, train, test):
    x1_col, x2_col = g.features
    x_lim = [min(torch.min(train.X[:,x1_col]).cpu(), torch.min(test.X[:,x1_col]).cpu()),max(torch.max(train.X[:,x1_col]).cpu(), torch.max(test.X[:,x1_col]).cpu())]
    y_lim = [min(torch.min(train.X[:,x2_col]).cpu(), torch.min(test.X[:,x2_col]).cpu()),max(torch.max(train.X[:,x2_col]).cpu(), torch.max(test.X[:,x2_col]).cpu())]
    
    x = np.arange(x_lim[0], x_lim[1], 0.1,dtype=np.float32)
    y = np.arange(y_lim[0], y_lim[1], 0.1,dtype=np.float32)
    
    X,Y = np.meshgrid(x,y)
    data = np.zeros((len(train.X[0]), len(X)*len(X[0])))
    print( f'data.shape {data.shape}')
    
    data[x1_col] = np.ravel(X)
    data[x2_col] = np.ravel(Y)
    
    data = data.T
    
    print(f'draw_truth {data.shape}')
    zs = np.array(g.func(data))
    Z = zs.reshape(X.shape)
    
    ax1.plot_surface(X,Y,Z,alpha=0.5,facecolor="#80800000")
    
    tx = train.X.cpu()
    ty = train.Y.cpu()
    
    ax1.scatter(tx[:,x1_col], tx[:,x2_col], ty, color="red")
    
    tx = test.X.cpu()
    ty = test.Y.cpu()
    
    ax1.scatter(tx[:,x1_col], tx[:,x2_col], ty, color="blue")
    
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("z")

def draw_model(ax1, g, nn, x_lim, y_lim):
    x1_col, x2_col = g.features    
    x = np.arange(x_lim[0], x_lim[1], 0.1, dtype=np.float32)
    y = np.arange(y_lim[0], y_lim[1], 0.1, dtype=np.float32)
    
    X,Y = np.meshgrid(x,y)
    print(f'draw_model layer_1 weights {nn.layer_1.weight.shape}')
    
    data = np.zeros((nn.layer_1.weight.shape[1], len(X)*len(X[0])),dtype=np.float32)
    print( f'draw_model data.shape {data.shape}')
    
    data[x1_col] = np.ravel(X)
    data[x2_col] = np.ravel(Y)
    
    data = torch.tensor(data).T.to(args.device)
    with torch.no_grad():
        z0 = nn(data)
    Zp = z0.reshape(X.shape).cpu()
    ax1.plot_surface(X,Y,Zp,facecolor="#80000080")
        
# def gen_func_sum(n_inputs, weights=None):
#     if (weights is None):
#         #sel=np.arange(n_inputs)
#         weights = torch.tensor([1.0 for i in range(n_inputs)])
#     print(f'gen_func_sum {n_inputs} {weights}')
#     return lambda x: np.float32(np.array(x).dot(np.array(weights)))

# def gen_func_product(n_inputs, weights=None):
#     if (weights is None):
#         #sel=np.arange(x_dim)
#         weights = torch.tensor([1.0 for i in range(n_inputs)])
#     print(f'gen_func_product {x_dim} {weights}')
#     def f(x):
#         p = [ w for w in weights if w != 0 else 1.0 ]

#         return p 
#     return f 

args = None

def main(argv = None):
    global args

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="nn_test")
    parser.add_argument("--hidden_dim", metavar="hidden_dim", type=int, default=1)
    parser.add_argument("--x_dim", "-x", metavar="x_dim", type=int, default=2)
    parser.add_argument("--i_dim", "-i", metavar="i_dim", type=int, default=0)
    parser.add_argument("--device", "-d", default=None)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0)
    
    parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument("--features", type=str, default = "")
    parser.add_argument("--value", type=float, default = None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--tensorboard", action="store_true", default=False)
    
    args = parser.parse_args(argv)
    
    total_dim = args.x_dim + args.i_dim

    if args.features != "":
        args.features = ",".split(args.features)
        if (len(args.features) != args.x_dim):
            logger.error(f'Number of relevant features {args.x_dim} does not match the number of features provided {args.features}')
    else:
        if (args.i_dim > 0):
            args.features = np.random.choice(np.arange(total_dim), args.x_dim, replace=False)
        else:
            args.features = np.arange(total_dim)
    
    # if args.value is not None:
    #     args.value = float(args.value)
        
    print(argv, args)
    if (args.device is None):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = FunctionGenerator(total_dim, args.features, args.value, lambda X: np.float32(np.sum(np.array(X), axis=-1)))
    
    train_data = Dataset(g, n_samples=args.n_train, noise=args.noise)
    
    t = np.array([[0,0], [0,1], [0.5,0.5], [0,1], [1,0], [0.5,0], [0,0.5],[1,1]])

    data = np.zeros((len(t),g.n_inputs),dtype=np.float32)
    for i,d in enumerate(t):
        for j,v in enumerate(d):
            data[i, g.features[j]] = v
    
    print(f't data {data.shape}')
    
    zt = g.func(data)
    
    print(t)
    print(data)
    print(zt)
    

    
    test_data = Dataset(g, args.n_test)
    print('train_data:',  train_data.X.shape, train_data.X.dtype, train_data.Y.shape, train_data.Y.dtype, 'test_data', test_data.X.shape, test_data.X.dtype, test_data.Y.shape, test_data.Y.dtype)
    
    model = NN(train_data.shape[1], args.hidden_dim, 1)
    model.to(args.device)
    
    print(model)
    
    vloss = 100000
    epochs = 0
    epoch_inc = 100
    
    tloss_s_x = []
    tloss_s = []
    
    vloss_s_x = []
    vloss_s_y = []

    writer = None
    if args.tensorboard:
        writer = tensorboard.SummaryWriter()
            
    while((epochs < args.max_epochs) and (vloss > 0.01)):
        tloss = model.train(train_data, epoch_inc, learning_rate=args.learning_rate, writer=writer)
        tloss_s.extend(tloss)
        
        vloss_s = model.test(test_data)
        #vloss_s = model.test(train_data)

        #print('vloss_s', len(vloss_s))
        vloss = sum(vloss_s)/len(vloss_s)
        vloss_s_x.append(epochs+epoch_inc)
        vloss_s_y.append(vloss)
        
        print( f'Epoch {epochs} Training loss {tloss[-1]} Validation loss {vloss}')
        epochs = epochs + epoch_inc                         
    
    if (len(g.features) == 2):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  
        # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    
        x1_col, x2_col = g.features
        x_lim = [min(torch.min(train_data.X[:,x1_col]).cpu(), torch.min(test_data.X[:,x1_col]).cpu()),
                 max(torch.max(train_data.X[:,x1_col]).cpu(), torch.max(test_data.X[:,x1_col]).cpu())]
        y_lim = [min(torch.min(train_data.X[:,x2_col]).cpu(), torch.min(test_data.X[:,x2_col]).cpu()),
                 max(torch.max(train_data.X[:,x2_col]).cpu(), torch.max(test_data.X[:,x2_col]).cpu())]

        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1, projection='3d')

        draw_truth(ax1, g, train_data, test_data)
        draw_model(ax1, g, model, x_lim, y_lim)
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.plot(tloss_s, label="train. loss")
        ax2.plot(vloss_s_x, vloss_s_y, "*-", label="valid. loss")
        ax2.legend()
        fig.savefig(f"./Figures/fig_{args.x_dim}_{args.i_dim}.png")
        plt.show()
    
    if writer:
        writer.close()
        
        
if __name__ == "__main__":
#    main(["--max_epochs", "1000", "--hidden_dim", "100", "--x_dim", "2", "--n_train", "100", "--n_test", "5", "--learning_rate", "0.01", "--tensorboard"])
    main(["--max_epochs", "1000", "--hidden_dim", "10", "--x_dim", "2", "--i_dim", "10", "--n_train", "10", "--n_test", "100", "--learning_rate", "0.01", "--value", "1.0", "--tensorboard"])
    