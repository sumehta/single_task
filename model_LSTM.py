# # Define the RNN Model
from torch.nn import functional
import torch.nn as nn
import torch
from torch.autograd import Variable


class LSTM( nn.Module ):

    def __init__( self, ntoken, ninp, nhid, nlayers, da, r, mlp_nhid, nclass, emb_matrix, cuda ):
        super( LSTM , self).__init__()
        """
        Args:
            ntoken:
            ninp:
            nhid:
            nlayers:
            da:
            r:
            mlp_nhid:
            nclass:
            emb_matrix:
            cuda:
        """


        # Embedding Layer
        self.encoder = nn.Embedding( ntoken, ninp )

        # RNN type
        self.rnn = nn.LSTM( ninp, nhid, nlayers, bias=False, batch_first=True, bidirectional=True )

        # # Self Attention Layers
        # self.S1 = nn.Linear( nhid * 2, da, bias=False )
        # self.S2 = nn.Linear( da, r, bias=False )

        # Final MLP Layers
        self.MLP = nn.Linear( nhid * 2, mlp_nhid )
        self.decoder = nn.Linear( mlp_nhid, nclass )

        self.init_wordembedding( emb_matrix )
        self.init_weights()

        self.r = r
        self.nhid = nhid
        self.nlayers = nlayers

        if cuda:
            self.cuda()

    def init_weights( self ):
        initrange = 0.1

        self.MLP.weight.data.uniform_( -initrange, initrange )
        self.MLP.bias.data.fill_( 0 )

        self.decoder.weight.data.uniform_( -initrange, initrange )
        self.decoder.bias.data.fill_( 0 )

    def init_wordembedding( self, embedding_matrix ):
        self.encoder.weight.data = embedding_matrix

    def forward(self, input, hidden, len_li ):
        emb = self.encoder( input )

        rnn_input = torch.nn.utils.rnn.pack_padded_sequence( emb, list( len_li.data ), batch_first=True )
        output, (hidden, cell) = self.rnn( rnn_input , hidden)

        # depacked_hidden, lens = torch.nn.utils.rnn.pad_packed_sequence( hidden, batch_first=True )

        if self.cuda:
            BM = Variable( torch.zeros( input.size( 0 ) , self.nhid * 2 ).cuda() )

        else:
            BM = Variable( torch.zeros( input.size( 0 ) , self.r * self.nhid * 2 ) )


        BM = hidden.view(input.size(0), self.nhid*2)

        # MLP block for Classifier Feature

        MLPhidden = self.MLP( BM )
        decoded = self.decoder( functional.relu( MLPhidden ) )

        return decoded

    def init_hidden( self, bsz ):
        weight = next( self.parameters() ).data

        return ( Variable( weight.new( self.nlayers * 2 , bsz, self.nhid ).zero_() ),
                Variable( weight.new( self.nlayers * 2, bsz, self.nhid ).zero_() ) )
