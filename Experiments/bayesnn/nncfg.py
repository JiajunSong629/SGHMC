"""
  Implementation of neutral network 
  network configurations
  Adapted from Tianqi Chen
  Edited by Jiajun Song, Yiping Song
"""
import nnet
import nnupdater
import numpy as np

class NNFactory:
    def __init__( self, param ):
        self.param = param
    def create_updater( self, w, g_w, sg_w ):
        return nnupdater.SGHMCUpdater( w, g_w, self.param )

    def create_hyperupdater( self, updaterlist ):
        return [ nnupdater.HyperUpdater( self.param, [u] ) for u in updaterlist ]
        
def softmax( param ):
    # setup network for softmax
    i_node = np.zeros( (param.batch_size, param.input_size), 'float32' )
    o_node = np.zeros( (param.batch_size, param.num_class), 'float32' )
    o_label = np.zeros((param.batch_size),'int8')

    nodes = [ i_node, o_node ]
    layers = [ nnet.FullLayer( i_node, o_node, param.init_sigma, param.rec_gsqr() )  ]

    layers+= [ nnet.SoftmaxLayer( o_node, o_label )]
    net = nnet.NNetwork( layers, nodes, o_label, factory ) 
    return net

def mlp2layer( param ):
    factory = NNFactory( param )
    # setup network for 2 layer perceptron
    i_node = np.zeros( (param.batch_size, param.input_size), 'float32' )
    o_node = np.zeros( (param.batch_size, param.num_class), 'float32' )
    h1_node = np.zeros( (param.batch_size, param.num_hidden), 'float32' )
    h2_node = np.zeros_like( h1_node )
    o_label = np.zeros((param.batch_size),'int8')

    nodes = [ i_node, h1_node, h2_node, o_node ]
    layers = [ nnet.FullLayer( i_node, h1_node, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ nnet.ActiveLayer( h1_node, h2_node, param.node_type )   ]
    layers+= [ nnet.FullLayer( h2_node, o_node, param.init_sigma, param.rec_gsqr() )  ]
    layers+= [ nnet.SoftmaxLayer( o_node, o_label )]

    net = nnet.NNetwork( layers, nodes, o_label, factory )    
    return net

# create a batch data from existing training data
# nbatch: batch size
# doshuffle: whether shuffle data first befor batch
# scale: scale the feature by scale
def create_batch( images, labels, nbatch, doshuffle=False, scale=1.0 ):
    if labels.shape[0] % nbatch != 0:
        print ('%d data will be dropped during batching' % (labels.shape[0] % nbatch))
    nsize = int(labels.shape[0] / nbatch * nbatch)
    assert images.shape[0] == labels.shape[0]

    if doshuffle:
        ind = list(range( images.shape[0] ))
        np.random.shuffle( ind )
        images, labels = images[ind], labels[ind]

    images = images[ 0 : nsize ]
    labels = labels[ 0 : nsize ]
    xdata = np.float32( images.reshape( labels.shape[0] // nbatch, nbatch, images[0].size ) ) * scale
    ylabel = labels.reshape( labels.shape[0] // nbatch, nbatch )    
    return xdata, ylabel
