#based on Hinton's MATLAB RBM code from <www.sciencemag.org/cgi/content/full/313/5786/504/DC1>
'''
This program trains Restricted Boltzmann Machine in which
visible, binary, stochastic pixels are connected to
hidden, tochastic real-valued feature detectors drawn from a unit
variance Gaussian whose mean is determined by the input from
the logistic visible units. Learning is done with 1-step Contrastive Divergence.
'''
import torch
class RBM():
    def __init__(self):
        self.epsilon_w = 0.1  # Learning rate for weights
        self.epsilon_v_b = 0.1  # Learning rate for biases of visible units
        self.epsilon_h_b = 0.1  # Learning rate for biases of hidden units
        self.weight_cost = 0.0002
        self.initial_momentum = 0.5
        self.final_momentum = 0.9
    def Train(self, batchdata, maxepoch=10,numhid = 8):
        (numcases, numdims, numbatches)=batchdata.shape

        # Initializing symmetric weights and biases.
        vishid = 0.1 * torch.randn(numdims, numhid)
        hidbiases = torch.zeros(1, numhid)
        visbiases = torch.zeros(1, numdims)

        vishidinc = torch.zeros(numdims, numhid)
        hidbiasinc = torch.zeros(1, numhid)
        visbiasinc = torch.zeros(1, numdims)
        batchposhidprobs = torch.zeros(numcases, numhid, numbatches)

        for epoch in range(maxepoch):
            print(1, 'epoch %d\r', epoch)
            errsum = 0
            for batch in range(numbatches):
                print(1, 'epoch %d batch %d\r', epoch, batch)
                # START POSITIVE PHASE
                data = batchdata[:,:, batch]
                temp_tensor1 = hidbiases.repeat(numcases, 1)
                temp_shape = temp_tensor1.shape
                poshidprobs = torch.ones(temp_shape) / (1 + torch.exp(-torch.matmul(data, vishid) - temp_tensor1))
                batchposhidprobs[:,:, batch]=poshidprobs
                posprods = torch.matmul(data.transpose(0, 1), poshidprobs)
                poshidact = torch.sum(poshidprobs, 0)
                posvisact = torch.sum(data, 0)

                # END OF POSITIVE PHASE
                poshidstates = (poshidprobs > torch.rand(numcases, numhid)).float()
                # START NEGATIVE PHASE
                temp_tensor2 = visbiases.repeat(numcases, 1)
                temp_shape = temp_tensor2.shape
                negdata = torch.ones(temp_shape) / (1 + torch.exp(-torch.matmul(poshidstates, vishid.transpose(0,1))
                                                                  - temp_tensor2))
                temp_shape = temp_tensor1.shape
                neghidprobs = torch.ones(temp_shape) / (1 + torch.exp(-torch.matmul(negdata, vishid) - temp_tensor1))
                negprods = torch.matmul(negdata.transpose(0,1), neghidprobs)
                neghidact = torch.sum(neghidprobs,0)
                negvisact = torch.sum(negdata,0)

                # END OF NEGATIVE PHASE
                err = sum(sum((data - negdata).pow(2)))
                errsum += err

                if epoch > 5:
                    momentum = self.final_momentum
                else:
                    momentum = self.initial_momentum

                # UPDATE WEIGHTS AND BIASES
                vishidinc = momentum * vishidinc + \
                            self.epsilon_w * ((posprods - negprods) / numcases - self.weight_cost * vishid)
                visbiasinc = momentum * visbiasinc + (self.epsilon_v_b / numcases) * (posvisact - negvisact)
                hidbiasinc = momentum * hidbiasinc + (self.epsilon_h_b / numcases) * (poshidact - neghidact)

                vishid = vishid + vishidinc
                visbiases = visbiases + visbiasinc
                hidbiases = hidbiases + hidbiasinc

                # END OF UPDATES

                print(1, 'epoch %4i error %6.1f  \n', epoch, errsum)
        return