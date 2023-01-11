import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter



def check_cosine(layer, gap):
    path = "/mnt/workspace/data/bloom/"
    x1 = np.memmap(path+f"h_{layer}.mmap", mode='r', shape=(500, 14336), dtype=np.float16)
    x1 = torch.tensor(x1).to('cuda')
    x2 = np.memmap(path+f"h_{layer+gap}.mmap", mode='r', shape=(500, 14336), dtype=np.float16)
    x2 = torch.tensor(x2).to('cuda')
    # return torch.mean(torch.nn.functional.cosine_similarity(x1, x2)), torch.std(torch.nn.functional.cosine_similarity(x1, x2))
    return torch.nn.functional.cosine_similarity(x1, x2)

def plot_dis(layer):
    path = "/mnt/workspace/data/bloom/"
    activation1 = np.memmap(path+f"4h_{layer}.mmap", mode='r', shape=(100, 14336*4), dtype=np.float16)
    activation1 = torch.tensor(activation1).to('cuda')
    return activation1.reshape(-1).cpu().numpy()


def pos_approx(act, w2, b2):
    # print('pos % ', torch.sum(act>0)/(act.shape[0]*act.shape[1]))
    x1_cp = act.clone()
    x1_cp[act<0] = 0
    x2_approx = torch.nn.functional.gelu(x1_cp) @ w2.t() + b2
    return x2_approx


def topk_approx(act, w2, b2, ratio=0.1):
    # _, indices = act.topk(k=int(act.shape[-1]*ratio), dim=-1)
    _, indices = torch.topk(act, k=int(act.shape[-1]*ratio), dim=-1)
    topk_mask = torch.zeros(act.shape[0], act.shape[1], dtype=int, device=act.device).scatter_(1, indices, 1).bool()
    x1_cp = torch.nn.functional.gelu(act)
    x1_cp[~topk_mask] = 0
    x2_approx_new = x1_cp @ w2.t() + b2
    return x2_approx_new


def pos_random_approx(act, w2, b2, ratio=0.1):
    x2_approx_new = torch.zeros((act.shape[0], w2.shape[0]), device=act.device)
    for i in range(act.shape[0]):
        indices = torch.zeros_like(act[i], dtype = int, device = 'cuda')
        ind1 = (act[i]>0)
        
        # random
        # ind2 = torch.rand((act.shape[1],), device=act.device) < 0.3
        # ind2 = torch.randperm(act.shape[1])
        # indices[ind2] = 1
        
        # mink
        samples = int(act.shape[1]*ratio)
        _, mask = torch.sort(torch.nn.functional.gelu(act[i]), descending=False)
        ind2 = torch.zeros((act.shape[1],), dtype=int, device=act.device).scatter_(0, mask[:samples], 1).bool()


        # _, mask = torch.topk(act[i], k=int(act.shape[1]*0.3), dim=-1)
        # ind2 = torch.zeros((act.shape[1],), dtype=int, device=act.device).scatter_(0, mask, 1).bool()

        ind2[ind1] = False
        prob = torch.sum(ind2) / (indices.shape[-1] - torch.sum(ind1))
        

        x2_approx_new[i, :] = torch.nn.functional.gelu(act[i][ind1]) @ (w2.t()[ind1]) + torch.nn.functional.gelu(act[i][ind2]) @ (w2.t()[ind2]) + b2
        # x2_approx_new[i, :] = torch.nn.functional.gelu(act[i][ind1]) @ (w2.t()[ind1]) + b2
    print(prob)
    
    return x2_approx_new


class LearnGeLU(nn.Module):
    def __init__(self, in_dim, out_dim, k=1000, ratio=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, k)
        self.fc2 = nn.Linear(k, out_dim)
        A = torch.ones((out_dim))/ratio.item()
        self.scale = nn.Parameter(A)
        self.topk = int(ratio*out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        logits = self.fc2(x)
        
        top_k_logits = torch.sigmoid(logits)*self.scale
        # top_k_logits = torch.softmax(logits, dim=-1) + self.scale
        top_k_gates, top_k_indices = top_k_logits.topk(self.topk, dim=1)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates


def learn_approx(x, x2, w1, b1, act, w2, b2, ratio=0.2, batch_size=100):
    # torch.Size([100, 14336]) torch.Size([100, 14336]) torch.Size([57344, 14336]) 
    # torch.Size([57344]) torch.Size([100, 57344]) torch.Size([14336, 57344]) torch.Size([14336])
    # print(x.shape, x2.shape, w1.shape, b1.shape, act.shape, w2.shape, b2.shape)    

    N = x.shape[0]
    num_batches = N // batch_size

    # training data: x, x2
    # network: 
    model = LearnGeLU(x.shape[-1], act.shape[-1], k=64, ratio=ratio).float().to(x.device)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1000, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0
        perm = torch.randperm(N)
        for i in range(num_batches):
            # get the inputs; data is a list of [inputs, labels]
            samples = perm[i*batch_size:(i+1)*batch_size]
            inputs = x[samples]
            labels = x2[samples]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            gates = model(inputs)
            
            # b x k  b x k x 
            pred = inputs@(w1.t())+b1
            gates[pred>0] = 0

            pos = pos_approx(pred, w2, b2)
            outputs = torch.nn.functional.gelu(pred) * gates @ w2.t() + b2

            loss = criterion(pos+outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 10 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss:.3f}')

    return model


def eval(model, inputs, w1, b1, w2, b2):
        # eval
    model.eval()
    with torch.no_grad():
        gates = model(inputs)
        pred = inputs@w1.t()+b1
        # gates[gates>0] = 1
        # gates[gates<0] = 0
        gates[pred>0] = 0
        x2_approx = torch.nn.functional.gelu(pred)*gates @ w2.t() + pos_approx(pred, w2, b2)
        # x2_approx = torch.nn.functional.gelu(pred)*gates @ w2.t() + b2
        # print('activated ratio ', ratio)
    return x2_approx


# class LearnGeLU(nn.Module):
#     def __init__(self, in_dim, out_dim, k=1000, ratio=0.1):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, k)
#         self.fc2 = nn.Linear(k, out_dim)
#         A = torch.zeros((out_dim))
#         self.scale = nn.Parameter(A)
#         self.topk = int(ratio*out_dim)
        
#     def forward(self, x):
#         x = self.fc1(x)
#         logits = self.fc2(x)
        
#         top_k_logits = torch.softmax(logits, dim=-1) + self.scale
#         # top_k_logits = torch.sigmoid(logits)
#         top_k_gates, top_k_indices = top_k_logits.topk(self.topk, dim=1)

#         zeros = torch.zeros_like(logits, requires_grad=True)
#         gates = zeros.scatter(1, top_k_indices, top_k_gates)
#         return gates


# def learn_approx(x, x2, w1, b1, act, w2, b2, ratio=0.2, batch_size=100):
#     # torch.Size([100, 14336]) torch.Size([100, 14336]) torch.Size([57344, 14336]) 
#     # torch.Size([57344]) torch.Size([100, 57344]) torch.Size([14336, 57344]) torch.Size([14336])
#     # print(x.shape, x2.shape, w1.shape, b1.shape, act.shape, w2.shape, b2.shape)    

#     N = x.shape[0]
#     num_batches = N // batch_size

#     # training data: x, x2
#     # network: 
#     model = LearnGeLU(x.shape[-1], act.shape[-1], k=64, ratio=ratio).float().to(x.device)
#     criterion = nn.MSELoss()
#     # optimizer = optim.SGD(model.parameters(), lr=1000, momentum=0.9)
#     optimizer = optim.AdamW(model.parameters(), lr=0.001)

#     for epoch in range(200):  # loop over the dataset multiple times
#         running_loss = 0.0
#         perm = torch.randperm(N)
#         for i in range(num_batches):
#             # get the inputs; data is a list of [inputs, labels]
#             samples = perm[i*batch_size:(i+1)*batch_size]
#             inputs = x[samples]
#             labels = x2[samples]
#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             gates = model(inputs)
            
#             # b x k  b x k x 
#             pred = inputs@(w1.t())+b1
#             gates[pred>0] = 0
#             # gates[gates>0] = 1

#             outputs = torch.nn.functional.gelu(pred)* gates

#             loss = criterion(outputs + torch.nn.functional.gelu(pred)*(pred>0).float(), torch.nn.functional.gelu(pred))
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#         if epoch % 10 == 0:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}] loss: {running_loss:.3f}')

#     return model


# def eval(model, inputs, w1, b1, w2, b2):
#         # eval
#     model.eval()
#     with torch.no_grad():
#         gates = model(inputs)
#         pred = inputs@w1.t()+b1
#         # gates[gates>0] = 1
#         # gates[gates<0] = 0
#         gates[pred>0] = 0
#         gates[gates>0] = 1
#         x2_approx = torch.nn.functional.gelu(pred)*gates @ w2.t() + pos_approx(pred, w2, b2)
#         # x2_approx = torch.nn.functional.gelu(pred*gates) @ w2.t() + b2
#         # print('activated ratio ', ratio)
#     return x2_approx


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, act='relu'):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GeLU()

    def load_weight(self, w1, b1, w2, b2):
        self.fc1.weight.data = w1
        self.fc2.weight.data = w2
        self.fc1.bias.data = b1
        self.fc2.bias.data = b2

    def forward(self, x, return_sparsity=False):
        x = self.act(self.fc1(x))
        logits = self.fc2(x)
        if return_sparsity:
            return logits, torch.mean((x>0).float())
        return logits


def distill_approx(x, x2, w1, b1, w2, b2, batch_size=128):
    # torch.Size([100, 14336]) torch.Size([100, 14336]) torch.Size([57344, 14336]) 
    # torch.Size([57344]) torch.Size([100, 57344]) torch.Size([14336, 57344]) torch.Size([14336])
    # print(x.shape, x2.shape, w1.shape, b1.shape, act.shape, w2.shape, b2.shape)    

    N = x.shape[0]
    num_batches = N // batch_size

    # training data: x, x2
    # network: 
    model = MLP(w1.shape[0], w1.shape[-1], act='relu').float().to(x.device)
    model.load_weight(w1, b1, w2, b2)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1000, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        perm = torch.randperm(N)
        for i in range(num_batches):
            # get the inputs; data is a list of [inputs, labels]
            samples = perm[i*batch_size:(i+1)*batch_size]
            inputs = x[samples]
            labels = x2[samples]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = model(inputs)
        
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 10 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss:.3f}')

    return model


def eval_distill(model, inputs):
        # eval
    model.eval()
    with torch.no_grad():
        logits, sparsity = model(inputs, return_sparsity=True)
    return logits, sparsity


def save_new_weights(layer, model, weight):
    path = "/mnt/workspace/checkpoint/bloom-relu/"

    w1 = model.fc1.weight.data.half()
    w2 = model.fc2.weight.data.half()
    b1 = model.fc1.bias.data.half()
    b2 = model.fc2.bias.data.half()

    weight['mlp.dense_h_to_4h.weight'] = w1
    weight['mlp.dense_4h_to_h.weight'] = w2
    weight['mlp.dense_h_to_4h.bias'] = b1
    weight['mlp.dense_4h_to_h.bias'] = b2
    torch.save(weight, path+f"pytorch_{layer}.pt")


def check_approx(layer):
    path = "/mnt/workspace/data/bloom/"
    x = np.memmap(path+f"h_{layer}.mmap", mode='r', shape=(2000, 14336), dtype=np.float16)
    x = torch.tensor(x).to('cuda').float()

    activation1 = np.memmap(path+f"4h_{layer}.mmap", mode='r', shape=(2000, 14336*4), dtype=np.float16)
    activation1 = torch.tensor(activation1).to('cuda').float()

    #torch.Size([100, 14336]) torch.Size([100, 57344])
    # print(x.shape, activation1.shape)

    path = "/mnt/workspace/checkpoint/bloom-new/"
    weight = torch.load(path+f"pytorch_{layer}.pt")

    w1 = weight['mlp.dense_h_to_4h.weight'].float().to('cuda')
    w2 = weight['mlp.dense_4h_to_h.weight'].float().to('cuda')
    b1 = weight['mlp.dense_h_to_4h.bias'].float().to('cuda')
    b2 = weight['mlp.dense_4h_to_h.bias'].float().to('cuda')

    # torch.Size([57344, 14336]) torch.Size([14336, 57344]) torch.Size([57344]) torch.Size([14336])
    # print(w1.shape, w2.shape, b1.shape, b2.shape)
    ratio = 0.3

    act = x@w1.t()+b1
    x2 = torch.nn.functional.gelu(act) @ w2.t() + b2
    # sanity check
    print('sanity check ', torch.norm(activation1-act), torch.norm(activation1))
    pos_ratio = torch.sum(act>0)/(act.shape[0]*act.shape[1])
    print('pos % ', pos_ratio)


    criterion = nn.MSELoss()
    # pos error
    x2_pos = pos_approx(act, w2, b2)
    # print('pos error ', torch.mean(torch.norm(x2-x2_pos, dim=-1)))
    print('pos error ', criterion(x2_pos, x2))


    # pos + random
    x2_pos_random = pos_random_approx(act, w2, b2, ratio=ratio-pos_ratio)
    print('pos+random error ', criterion(x2_pos_random, x2))

    # topk
    x2_topk = topk_approx(act, w2, b2, ratio=ratio)
    # print('topk error ',torch.mean(torch.norm(x2-x2_topk, dim=-1)))
    print('topk error ', criterion(x2_topk, x2))


    train_data = x[:1800]
    train_labels = x2[:1800]
    
    eval_data = x[1800:]
    eval_labels = x2[1800:]
    
    # model = learn_approx(train_data, train_labels, w1, b1, act, w2, b2, ratio=ratio-pos_ratio, batch_size=200)

    # x2_learn = eval(model, eval_data, w1, b1, w2, b2)
    # # print('learn error ', torch.mean(torch.norm(eval_labels-x2_learn, dim=-1)))
    # print('learn error ', criterion(x2_learn, eval_labels))

    model = distill_approx(train_data, train_labels, w1, b1, w2, b2, batch_size=100)
    x2_distill, sparsity = eval_distill(model, eval_data)
    print('distill error ', criterion(x2_distill, eval_labels), sparsity)

    save_new_weights(layer, model, weight)

    # x2_distill, sparsity = eval_distill(model, train_data)
    # print('learn error ', torch.mean(torch.norm(train_labels-x2_distill, dim=-1)), sparsity)



means = torch.zeros((69, 500))
# images = 70
# fig, axs = plt.subplots(7, 10, sharey=True, tight_layout=True)
# fig.set_size_inches(8*10, 4*7)
# n_bins = 100

# for layer in range(0, 10):
# for layer in range(10, 20):
# for layer in range(20, 30):
# for layer in range(30, 40):
# for layer in range(40, 50):
# for layer in range(50, 60):
for gap in range(1, 2):
    for layer in range(0, 69):
        print('==========================================')
        print(f'layer {layer} \n')

        # check_approx(layer)

        # plot input dis
        # dist1 = plot_dis(layer)
        # axs[layer//10, layer%10].hist(dist1, bins=n_bins)
        mean = check_cosine(layer, gap)
        means[layer, :] = mean

        # print(mean, std)
        print('==========================================')
        print('\n', flush=True)

print(means[:, 300])
#     plt.plot(list(range(66)), means[:, gap-1], label = str(gap))
# print(means)
# plt.legend()
# plt.savefig('cosine.png')