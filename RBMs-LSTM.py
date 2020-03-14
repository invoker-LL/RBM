'''
Train/test in MNIST
'''
import torch
import torchvision
import torchvision.transforms as transforms
import RBM
import torch.nn as nn
class RBMLSTMModel(nn.Module):
    def __init__(self, ninput, n_rbm_unit, nhid, nlayers, noutput, dropout=0.25):
        super(RBMLSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(ninput, n_rbm_unit)
        self.rnn = nn.LSTM(n_rbm_unit, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, noutput)

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange) # todo,weights_from_rbm
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        z = self.drop(self.encoder(input))
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
def main():
    # train on the GPU or on the CPU, if a GPU is not available
    batch_size = 16
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # get MNIST data, and shuffle them
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    #rbm = RBM.RBM()
    #weights = rbm.Train(trainloader, numhid=28)
    #use this weights to update first dense
    model = RBMLSTMModel(
        noutput=10,
        ninput=28,
        n_rbm_unit=32,
        nhid=64,
        nlayers=1,
    )

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    def train(inputdata,labels):
        # move model\data\hidden_states to the right device
        hidden = model.init_hidden(batch_size, cuda_flag=True)
        inputdata = inputdata.to(device)
        labels = labels.to(device)
        model.to(device)

        model.zero_grad()
        output = model(inputdata, hidden)
        loss = criterion(output[:,-1,:], labels)
        loss.backward()
        optimizer.step()

        return loss.item()
    def test():
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                hidden = model.init_hidden(batch_size, cuda_flag=True)
                images, labels = data
                temp_shape = images.shape
                images = images.reshape([batch_size, temp_shape[2], temp_shape[3]])
                images = images.to(device)
                labels = labels.to(device)
                model.to(device)
                outputs = model(images,hidden)
                _, predicted = torch.max(outputs[:,-1,:], 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        sum_acc = 0
        for i in range(10):
            print('Accuracy of %3s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            sum_acc += class_correct[i] / class_total[i] * 100

        print('Accuracy of all: %.2f %%' % (sum_acc/10))

    n_epoch = 5
    for iter in range(n_epoch):
        for i, data in enumerate(trainloader, 0): # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            temp_shape = inputs.shape
            inputs = inputs.reshape([batch_size, temp_shape[2], temp_shape[3]])
            loss = train(inputs, labels)
            if i % 100 == 1:
                print('loss:', loss)
    print('Finished Training')

    torch.save(model.state_dict(), 'lastParams')

    model.load_state_dict(torch.load('lastParams'))
    model.eval()
    test()
if __name__ == "__main__":
    main()