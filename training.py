import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from load_data import load_dataset
from models.LSTM_model import LSTM # general LSTM 
from models.LSTM_attention_model import LSTM_attn # LSTM with attention

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda() # use GPU
    # Adam algorithm for optimization
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001) 
    model.train()
    steps = 0
    for idx, batch in enumerate(train_iter):
        # input features: concat comment and parent_comment together
        ### text = batch.comment[0]
        text = torch.cat((batch.comment[0], batch.parent_comment[0]), dim=1)
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 8):# One of the batch returned by BucketIterator has length different than 8.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        '''
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        '''
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            ### text = batch.comment[0]
            text = torch.cat((batch.comment[0], batch.parent_comment[0]), dim=1)
            if (text.size()[0] is not 8):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

# load datasets
dataset_path = 'C:\\Urim_Thummim\\Rutgers\\Fall 2019\\Principles of AI\\Final Project\\dataset\\'
TEXT, train_iter, valid_iter, test_iter = load_dataset(dataset_path)

# parameters 
batch_size = 8
output_size = 2 # 0(neg) or 1(pos)
hidden_size = 256
vocab_size = len(TEXT.vocab)
embedding_length = 300
word_embeddings = TEXT.vocab.vectors # weights of the LSTM network

# learning rate and loss function
loss_fn = F.cross_entropy

# initializing LSTM model
model = LSTM_attn(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
### model = LSTM(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

# training and testing
for epoch in range(5):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    ### val_loss, val_acc, val_rec, val_pre, val_f1 = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = eval_model(model, test_iter)
### test_loss, test_acc, test_rec, test_pre, test_f1 = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
### print(f'Test Recall: {test_rec:.2f}%, Test Precision: {test_pre:.2f}%, Test F1 Score: {test_f1:.2f}%')