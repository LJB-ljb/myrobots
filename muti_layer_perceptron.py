import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import csv
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


class MLP(nn.Module):
    """build the MLP network"""
    def __init__(self, learning_rate=0.01):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            # input layer
            nn.Linear(57,200),
            nn.ReLU(inplace= True),
            # hidden layer
            nn.Linear(200, 200),
            nn.ReLU(inplace= True),
            # output layer
            nn.Linear(200, 2),
            nn.ReLU(inplace= True)
        )

        # using CrossEntropyLoss and Adam
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.model(x)
        return x

    def train(self, x, label):
        out = self.forward(x)
        
        loss = self.mse(out, label)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()




def read_data(csvfile):
    """read datas from file using csv format"""
    data = []
    results = []
    with open(csvfile) as f:
        csv_reader = csv.reader(f)
        [data.append(row) for row in csv_reader]

    data.remove(data[0])
    for d in data:   
        for i in range(len(d)):
            d[i] = float(d[i])
        results.append(d[-1])
        d.remove(d[-1])         
    return np.array(data), np.array(results)

def accuracy(pred,real):
    """caculate the accuracy """
    pred_cls = torch.argmax(nn.Softmax(dim=1)(pred),dim=1).data
    return accuracy_score(real,pred_cls)

def F1Score(pred, real):
    """caculate the F1 score"""
    pred_cls = torch.argmax(nn.Softmax(dim=1)(pred),dim=1).data
    return accuracy_score(real,pred_cls)





def train():
    """train the moddel"""
    # hyperparmeter
    batch_size = 100
    learning_rate = 0.01
    epochs=100
    CUDA = torch.cuda.is_available()

    mlp = MLP(learning_rate)



    if CUDA:
        # if GPU available, using GPU
        mlp.cuda()

    train, train_results = read_data("/home/ljb/Machine_Learning/D02-multi-layer-perceptron/training_data.CSV")
    train_dataset = Data.TensorDataset(torch.from_numpy(train).float(),
                                       torch.from_numpy(train_results).long())
    train_loader = Data.DataLoader(dataset= train_dataset, 
                                   batch_size= batch_size, shuffle= True)
    dfhistory = pd.DataFrame(columns = ["epoch","loss","accuracy", "F1"]) 

    print("===Start Training===")

    for epoch in range(epochs):
        loss_sum = 0.0
        metric_sum = 0.0

        for step, (d, r) in enumerate(train_loader):
            r = torch.reshape(r, [batch_size])

            out = mlp.forward(d)
            loss = mlp.mse(out, r)
            metric = accuracy(out, r)
            F1 = F1Score(out, r)
            mlp.optim.zero_grad()
            loss.backward()
            mlp.optim.step()

            loss_sum =  loss_sum + loss.item()
            metric_sum = metric_sum + metric.item()
            print("Epoch: {} | Step: {} | loss: {} | accuracy: {} | F1-Score: {}"
                  .format(epoch, step+1, loss_sum/(step+1), metric_sum/(step+1), F1))
        
        dfhistory.loc[epoch] = (epoch, loss_sum/(step+1), metric_sum/(step+1), F1)
    
    print("===Finish Training===")
    torch.save(mlp.state_dict(), "weight.pkl")
    return dfhistory

    
def predict():
    """predict"""
    test, test_results = read_data("/home/ljb/Machine_Learning/D02-multi-layer-perceptron/testing_data.CSV")
    test_dataset = Data.TensorDataset(torch.from_numpy(test).float(),
                                       torch.from_numpy(test_results).long())
    test_loader = Data.DataLoader(dataset= test_dataset, shuffle= True)
    CUDA = torch.cuda.is_available()
    mlp = MLP()
    mlp.load_state_dict(torch.load("weight.pkl"))
    if CUDA:
        mlp.cuda()
    
    acc_sum = 0
    for step, (d, r) in enumerate(test_loader):
        out = mlp.forward(d)
        acc = accuracy(out, r)
        acc_sum = acc_sum + acc
        # if step %
    print("The accuracy rate is : {}.".format(acc_sum/len(test_loader))) 


def plot_results(dfhistory):
    """plot the results of loss and accuracy"""
    acc = dfhistory["accuracy"]
    loss = dfhistory["loss"]
    
    plt.figure(1)
    plt.plot(range(1, len(acc)+1), acc, 'r-')
    plt.title('Training Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("training_accuracy.png")
    plt.show()

    
    plt.figure(2)
    plt.plot(range(1, len(loss)+1), loss, 'b--')
    plt.title('Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":

    dfhistory = train()
    plot_results(dfhistory)
    # predict()