
def train(net, modelName,optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    output,loss_CI,_,_ = net(data.x, data.adj)

    loss =criterion(output[data.train_mask], data.y[data.train_mask])
    loss+=loss_CI

    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc,0,0

def val_and_test(net, modelName,data):
    net.eval()
    output,loss_CI,_,_  = net(data.x, data.adj)

    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return  acc_test,acc_val,0,0

def accuracy(output, labels):
    preds = output.max(1)[1]
    preds=preds.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



