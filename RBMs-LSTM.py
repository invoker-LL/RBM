'''
Train/test in MNIST
'''
import torch
import torchvision
import torchvision.transforms as transforms
import RBM
import torch.nn as nn
class RBMLSTMModel(nn.Module):
    def __init__(self, ninput, n_rbm_unit, nhid, nlayers, noutput, dropout=0.25, RBM_weights = []):
        super(RBMLSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(ninput*ninput, ninput * n_rbm_unit)
        self.rnn = nn.LSTM(n_rbm_unit, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, noutput)

        self.ninput = ninput
        self.n_rbm_unit = n_rbm_unit
        self.nlayers = nlayers
        self.nhid = nhid

        self.init_weights(RBM_weights)

    def init_weights(self,RBM_weights):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if RBM_weights == []: # 1. default param init
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.encoder.bias.data.zero_()
        else: # 2. init param with RBM weights!
            self.encoder.weight.data = RBM_weights[0]
            self.encoder.bias.data = RBM_weights[1]

    def forward(self, input, hidden):
        z = self.encoder(input.reshape(-1, self.ninput * self.n_rbm_unit))
        z = z.reshape(-1, self.ninput, self.n_rbm_unit)
        output, hidden = self.rnn(z, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded

    def init_hidden(self, bsz, cuda_flag = False):
        weight = next(self.parameters())
        if cuda_flag == False:
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
        return (weight.new_zeros(self.nlayers, bsz, self.nhid).cuda(),
                weight.new_zeros(self.nlayers, bsz, self.nhid).cuda())

def train(model,trainloader,optimizer,device,criterion):
    for i, data in enumerate(trainloader, 0):  # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        temp_shape = images.shape
        images = images.reshape([temp_shape[0], temp_shape[2], temp_shape[3]])
        images = images.to(device)
        labels = labels.to(device)
        hidden = model.init_hidden(temp_shape[0], cuda_flag=True)
        model.to(device)

        model.zero_grad()
        output = model(images, hidden)
        loss = criterion(output[:, -1, :], labels)
        loss.backward()
        optimizer.step()

    print('loss:', loss.item())

def test(model,testloader,device,classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            temp_shape = images.shape
            hidden = model.init_hidden(temp_shape[0], cuda_flag=True)
            images = images.reshape([temp_shape[0], temp_shape[2], temp_shape[3]])
            images = images.to(device)
            labels = labels.to(device)
            model.to(device)
            outputs = model(images, hidden)
            _, predicted = torch.max(outputs[:, -1, :], 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    sum_acc = 0
    eps = 0.00000001
    for i in range(10):
        print('Accuracy of %3s : %2d %%' % (
            classes[i], 100 * class_correct[i] / (class_total[i] + eps)))
        sum_acc += class_correct[i] / (class_total[i] + eps) * 100

    print('Accuracy of all: %.2f %%' % (sum_acc / 10))

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get MNIST data, and shuffle them
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(trainset)).tolist()
    trainset = torch.utils.data.Subset(trainset, indices[:3000])

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    indices = torch.randperm(len(testset)).tolist()
    testset = torch.utils.data.Subset(testset, indices[:])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # we will use RBM trained weights to init first encoder layer
    #rbm = RBM.RBM().to(device)
    #RBM_weights = rbm.RBMtrain(trainloader, numdims=28*28, numhid=28*28, maxepoch=2)

    model = RBMLSTMModel(
        noutput=10,
        ninput=28,
        n_rbm_unit=28,
        nhid=64,
        nlayers=1,
        RBM_weights=[]
    )

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    n_epoch = 20
    for iter in range(n_epoch):
        train(model,trainloader,optimizer,device,criterion)

    torch.save(model.state_dict(), 'lastParams')
    model.load_state_dict(torch.load('lastParams'))

    model.eval()
    test(model,testloader,device,classes)

if __name__ == "__main__":
    main()