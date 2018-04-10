# Utility.py
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Batchify the whole dataset
def select_data( data, bsz ):
    try:
        nbatch = data.size( 0 ) // bsz
        data = data.narrow( 0, 0, nbatch * bsz )

    except:
        nbatch = len( data ) // bsz
        data = data[ : nbatch * bsz ]
        
    return data


# Unpack the hidden state
def repackage_hidden( h ):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type( h ) == Variable:
        return Variable( h.data )
    else:
        return tuple( repackage_hidden( v ) for v in h )


# Retrieve a batch from the source
def get_batch( source, labels, len_list, i, size, cuda, evaluation=False ):
    batch_size = size
    data = source[ i : i + batch_size ]
    labels = labels[ i : i + batch_size ]
    len_li = len_list[ i : i + batch_size ]

    # sort descending
    a = np.argsort(len_li)
    a = a[::-1]

    data = data.cpu().numpy()[a]
    data = torch.from_numpy(data)

    labels = labels.cpu().numpy()[a]
    labels = torch.from_numpy(labels)

    len_li = sorted(len_li)
    len_li = len_li[::-1]

    if cuda:
        data = Variable( data.cuda() , volatile = evaluation )
        # target = Variable( labels[ i : i + batch_size ].view( -1 ).cuda() )
        target = Variable( labels.cuda() )  #remove view for multi-task
        len_li = Variable( torch.LongTensor( len_li  ).cuda() )
    else:
        data = Variable( data , volatile = evaluation )
        # target = Variable( labels[ i : i + batch_size ].view( -1 ) )
        target = Variable( labels )
        len_li = Variable( torch.LongTensor( len_li) )

    return data, target, len_li



# Function to compute precision, recall, f1 and accuracy
def compute_measure( pred, target ):
    pre = []
    rec = []
    f1 = []
    pre.append(precision_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels =[0, 1], average='micro'))
    pre.append(precision_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels =[0, 1], average='macro'))

    rec.append(recall_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels=[0, 1], average='micro'))
    rec.append(recall_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels=[0, 1], average='macro'))

    f1.append(f1_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels=[0, 1], average='micro'))
    f1.append(f1_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0], labels=[0, 1], average='macro'))
               
    acc = accuracy_score(pred.data.cpu().numpy()[0], target.data.cpu().numpy()[0],  normalize=True)
    return pre, rec, f1, acc


# Get Attention Weights Function
def Attentive_weights( model, data_source, labels, data_len, eval_batch_size, cuda ):
    
    hidden = model.init_hidden( eval_batch_size )
    Weights = {}
    
    for i in range( 0, data_source.size( 0 ) - 1, eval_batch_size ):
        
        data, targets, len_li = get_batch( data_source, labels, data_len, i, eval_batch_size, cuda, evaluation=True )
        output, _, _, weights = model( data, hidden, len_li )
        
        Weights[ i ] = [ weights, output ]

    return Weights
