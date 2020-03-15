import torch
import torch.nn as nn
class RBM(nn.Module):
    def __init__(self):
        super(RBM, self).__init__()
        self.epsilon_w = 0.1  # Learning rate for weights
        self.epsilon_v_b = 0.1  # Learning rate for biases of visible units
        self.epsilon_h_b = 0.1  # Learning rate for biases of hidden units
        self.weight_cost = 0.0002
        self.initial_momentum = 0.5
        self.final_momentum = 0.9

    def RBMtrain(self, data_loader, maxepoch=5, numdims=28,numhid=32):
        # Initializing symmetric weights and biases.
        v_h_w = 0.1 * torch.randn(numdims, numhid)
        h_bias = torch.zeros(1, numhid)
        v_bias = torch.zeros(1, numdims)

        v_h_w_inc = torch.zeros(numdims, numhid)
        h_bias_inc = torch.zeros(1, numhid)
        v_bias_inc = torch.zeros(1, numdims)

        for epoch in range(maxepoch):
            err_sum = 0
            cnt = 0
            for batchdata in data_loader:
                data, labels = batchdata
                temp_shape = data.shape
                data = data.reshape([temp_shape[0], -1])

                (batchsize, _) = data.shape
                temp_tensor1 = h_bias.repeat(batchsize, 1)
                temp_shape = temp_tensor1.shape
                poshidprobs = torch.ones(temp_shape) / (
                        1 + torch.exp(-torch.matmul(data, v_h_w) - temp_tensor1))
                posprods = torch.matmul(data.transpose(0, 1), poshidprobs)
                poshidact = torch.sum(poshidprobs, 0)
                posvisact = torch.sum(data, 0)

                # END OF POSITIVE PHASE
                poshidstates = (poshidprobs > torch.rand(batchsize, numhid)).float()
                # START NEGATIVE PHASE
                temp_tensor2 = v_bias.repeat(batchsize, 1)
                temp_shape = temp_tensor2.shape
                negdata = torch.ones(temp_shape) / (
                        1 + torch.exp(-torch.matmul(poshidstates, v_h_w.transpose(0, 1))
                                      - temp_tensor2))
                temp_shape = temp_tensor1.shape
                neghidprobs = torch.ones(temp_shape) / (
                        1 + torch.exp(-torch.matmul(negdata, v_h_w) - temp_tensor1))
                negprods = torch.matmul(negdata.transpose(0, 1), neghidprobs)
                neghidact = torch.sum(neghidprobs, 0)
                negvisact = torch.sum(negdata, 0)

                # END OF NEGATIVE PHASE
                err = sum(sum((abs(data - negdata))))/batchsize
                cnt += 1
                err_sum += err
                if epoch > 5:
                    momentum = self.final_momentum
                else:
                    momentum = self.initial_momentum

                # UPDATE WEIGHTS AND BIASES
                v_h_w_inc = momentum * v_h_w_inc + \
                            self.epsilon_w * ((posprods - negprods) / batchsize - self.weight_cost * v_h_w)
                v_bias_inc = momentum * v_bias_inc + (self.epsilon_v_b / batchsize) * (posvisact - negvisact)
                h_bias_inc = momentum * h_bias_inc + (self.epsilon_h_b / batchsize) * (poshidact - neghidact)

                v_h_w = v_h_w + v_h_w_inc
                v_bias = v_bias + v_bias_inc
                h_bias = h_bias + h_bias_inc

            print('epoch: %d, error %.1f ' % (epoch, err_sum/cnt))

        return v_h_w, h_bias