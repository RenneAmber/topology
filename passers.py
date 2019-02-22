from utils import progress_bar
import torch
import numpy as np
from labels import identity


def get_accuracy(predictions, targets):
    ''' Compute accuracy of predictions to targets. max(predictions) is best'''
    _, predicted = predictions.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return 100.*correct/total


class Passer():
    def __init__(self, net, loader, criterion, device, repeat=1):
        self.network = net
        self.criterion = criterion
        self.device = device
        self.loader = loader
        self.repeat = repeat

    def _pass(self, optimizer=None, manipulator=identity):
        ''' Main data passing routing '''
        losses, features, total, correct = [], [], 0, 0
        accuracies = []
        
        for r in range(1, self.repeat+1):
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                # print("[passer _pass] target before shape", targets.shape,"input shape", inputs.shape)
                targets = manipulator(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                if optimizer: optimizer.zero_grad()
                outputs = self.network(inputs)
                # print("[passer _pass] outputs shape", outputs.shape)
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())
            
                if optimizer:
                    loss.backward()
                    optimizer.step()

                accuracies.append(get_accuracy(outputs, targets))
                progress_bar((r-1)*len(self.loader)+batch_idx, r*len(self.loader), 'repeat %d -- Mean Loss: %.3f | Last Loss: %.3f | Acc: %.3f%%'
                             % (r, np.mean(losses), losses[-1], np.mean(accuracies)))

        return np.asarray(losses), np.mean(accuracies)
    
    
    def run(self, optimizer=None, manipulator=identity):
        if optimizer:
            self.network.train()
            return self._pass(optimizer, manipulator=manipulator)
        else:
            self.network.eval()
            with torch.no_grad():
                return self._pass(manipulator=manipulator)

            
    def get_function(self):
        ''' Collect function (features) from the self.network.module.forward_features() routine '''
        features = []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)
                
            features.append([f.cpu().data.numpy().astype(np.float16) for f in self.network.module.forward_features(inputs)])

            progress_bar(batch_idx, len(self.loader))

        return [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]

    def get_structure(self):
        ''' Collect structure (weights) from the self.network.module.forward_weights() routine '''
        # modified #
        ## NOTICE: only weights are maintained and combined into two dimensions, biases are ignored
        weights = []
        weight_save = []
        #[print("we get data type is {}, size is {}".format(type(f.data),f.size())) for f in self.network.parameters()]
        for index, var in enumerate(self.network.parameters()):
            if index % 2 == 0:
                f = var.cpu().data.numpy().astype(np.float16) # var as Variable, type(var.data) is Tensor, should be transformed from cuda to cpu(),with type float16
                weight = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))
                print("weight size ==== ", weight.shape)
                weights.append(weight)
        for li,w in enumerate(weights):
            weight_save.append({'layer':li+1,'mean':np.mean(w), 'std':np.std(w), 'min': w.min(), 'max': w.max()})

        return weights, weight_save

    def get_structure_layer(self,layer=1):
        # Check the weight we get and generate the gabor filter
        # Layer 1 correlation 
        var1 = list(self.network.parameters())[2*(layer-1)]
        f = var1.cpu().data.numpy().astype(np.float16) # var1 as first layer Variable, type(var.data) is Tensor, should be transformed from cuda to cpu(),with type float16
        l = len(f.shape)
        if l > 2:        
            weight = np.reshape(f, (f.shape[0]*f.shape[1], np.prod(f.shape[2:])))
        else:
            weight = f
        return weight

    def get_network_structure(self):
        # Get layer information 
        # e.g.: for conv2d: we get {input, output, stride, padding, kernel_size, weights}
        #       for fc, we get {in, out, weights}
        # print('---\n',self.loader)
        layers = self.network.named_modules()
        layer_info_dict = {}
        for name, layer in layers:
            tl = type(layer)
            if tl == type(torch.nn.Conv2d(1,1,3)):
                param = [p for p in layer.parameters()][0].data.cpu().numpy()
                layer_info_dict[name] = {'name':'conv2d', 'in':layer.in_channels,'out':layer.out_channels, 'stride':layer.stride,'padding':layer.padding, 'ks':layer.kernel_size, 'weights':param}
            elif tl == type(torch.nn.Linear(2,1)):
                param = [p for p in layer.parameters()][0].data.cpu().numpy()
                layer_info_dict[name] = {'name':'fc', 'in':layer.in_features,'out':layer.out_features, 'weights':param}
            else:
                pass
                # print("Undefined",type(layer))
        return layer_info_dict
'''

def train(net, trainloader, device, optimizer, criterion, do_optimization, shuffle_labels, n_batches):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_acc = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if n_batches and batch_idx > n_batches: break
            
        if shuffle_labels and list(targets.size())[0]==128:
            targets = targets[PERM]
                
        inputs, targets = inputs.to(device), targets.to(device)
                
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss_acc.append(loss.item())

        loss.backward()
        optimizer.step()
                
        train_loss += loss.item()
        _, predicted = outputs.max(1)
                
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.*correct/total
    
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return np.asarray(loss_acc), accuracy

def train_subset(net, trainloader, device, optimizer, criterion, do_optimization, shuffle_labels, n_batches):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_acc = []
    if do_optimization:
        n_repeats = int((60000/128)/n_batches)
    else:
        n_repeats = 1

    
    for repeat in range(0, n_repeats):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if n_batches and batch_idx>n_batches: break
            
            if shuffle_labels and list(targets.size())[0]==128:
                targets = targets[PERM]
                
            inputs, targets = inputs.to(device), targets.to(device)
                
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss_acc.append(loss.item())

            if do_optimization:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
                
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.*correct/total
    
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return np.asarray(loss_acc), accuracy


def test(net, testloader, device, criterion, n_test_batches):
    net.eval()
    test_loss, correct, total, target_acc, activation_acc = 0, 0, 0, [], []
    loss_acc = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):            
            inputs, targets = inputs.to(device), targets.to(device)            
            outputs = net(inputs)

            if batch_idx < n_test_batches:
                activations = [a.cpu().data.numpy().astype(np.float16) for a in net.module.forward_features(inputs)]
                target_acc.append(targets.cpu().data.numpy())
                activation_acc.append(activations)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            loss_acc.append(loss.item())
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total
                 
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(batch_idx+1), accuracy))
          
    activs = [np.concatenate(list(zip(*activation_acc))[i]) for i in range(len(activation_acc[0]))]
    
    return (activs, np.concatenate(target_acc), np.asarray(loss_acc), accuracy)
'''
