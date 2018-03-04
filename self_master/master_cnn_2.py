import numpy as np
import math
import os

''' LeNet 
input 784 -> Convolution1 -> Convolution2 -> HiddenLayer 100 -> output 10,
without max pooling
    Convolution weight to zero
node weight
0 - 0 1 2 28 29 30 56 57 58
1 - 1 2 3 29 30 31 57 58 59

    0 1 2 28 29 30 56 57 58
...
26 - 28 29 30 56 57 58 84 85 86
27 - 29 30 31 57 58 59 85 86 87

     2  3  4  30 31 32 58 59 60
...
52 - 56 57 56 84 85 86 112 113 114
53 - 57 58 59 85 86 87 113 114 115

     4  5  6  32 33 34 60  61  62
'''


class Model( object ):

    def __init__( self, kernel_size1 = [ 5, 5 ], num_of_filter = 8 ):
        # define variable 
        self.input_layer = None
        self.labels = None
        self.Nodes = { 'input_layer': 784, 'labels': 10, 'Convolution_layer1': None, 'layer1': 2000, 'layer2': 10 }

        self.Convolution_layer1 = { 'weights': None, 'output': None, 'delta': None, 'num_of_filter': None, 'kernel_size': None, 'init': None, 'final': None }
        self.layer1 = { 'weights': None, 'prediction': None, 'delta': None }
        self.layer2 = { 'weights': None, 'prediction': None, 'delta': None }

        self.final = 0

        # variable __init__
        self.input_layer = np.random.random(( 1, self.Nodes['input_layer'] ))
        self.labels = np.random.random(( 1, self.Nodes['labels'] ))
        # Convolution
        self.Convolution_init( kernel_size1, num_of_filter )
        # Fully connected layer
        self.layer1['weights'] = 2 * np.random.random(( self.Nodes['Convolution_layer1'], self.Nodes['layer1'] )) - 1
        self.layer1['prediction'] = np.random.random(( 1, self.Nodes['layer1'] ))
        self.layer2['weights'] = 2 * np.random.random(( self.Nodes['layer1'], self.Nodes['layer2'] )) - 1
        self.layer2['prediction'] = np.random.random(( 1, self.Nodes['layer2'] ))


    def Convolution_init( self, kernel_size1, num_of_filter ):
        self.Convolution_layer1['num_of_filter'] = num_of_filter
        self.Convolution_layer1['kernel_size'] = kernel_size1
        size_of_output = ( 28 - kernel_size1[0] + 1 ) ** 2
        # define convolution layer output nodes
        self.Nodes['Convolution_layer1'] = size_of_output * num_of_filter
        self.Convolution_layer1['weights'] = 2 * np.random.random(( self.Nodes['input_layer'], self.Nodes['Convolution_layer1'] )) - 1
        self.Convolution_layer1['output'] = np.random.random(( 1, self.Nodes['Convolution_layer1'] ))
        # each node has its input node else equal 0
        # tmp[0], node 0 and its input node
        first = 0
        init = []
        size_of_output = 28 - kernel_size1[0] + 1
        for i in range( kernel_size1[0] ):
            for k in range( kernel_size1[0] ):
                init.append( k + first )
            first += 28
        first = 0
        tmp = []
        for n in range( size_of_output ):
            for i in range( size_of_output ):
                tmp2 = []
                for k in init:
                    tmp2.append( k + i + size_of_output * n + first )
                tmp.append( tmp2 )
            first += 2
        final = tmp * num_of_filter
        self.Convolution_layer1['init'] = tmp
        self.Convolution_layer1['final'] = final
        # set weight to zero
        for i in range( self.Nodes['Convolution_layer1'] ):
            for k in range(784):
                if k not in final[i]:
                    self.Convolution_layer1['weights'][k][i] = 0

    def Convolution_Reconstruct( self ):
        # weights to zero
        for i in range( self.Nodes['Convolution_layer1'] ):
            for k in range(784):
                if k not in self.Convolution_layer1['final'][i]:
                    self.Convolution_layer1['weights'][k][i] = 0
        # share weights
        num_of_each_kernel_weights = self.Convolution_layer1['kernel_size'][0] * self.Convolution_layer1['kernel_size'][1]
        for i in range( self.Convolution_layer1['num_of_filter'] ): # 8
            for k in range( num_of_each_kernel_weights ):
                # in each init filter has kernel_size * kernel_size filter weightsm in this case is 5X5
                weights_avg = 0
                for weights_in_filter in range( len( self.Convolution_layer1['init'] ) ): # 24 * 24
                    Processing_Node = ( i + 1 ) * ( weights_in_filter + 1 ) - 1
                    input_layer_node = self.Convolution_layer1['init'][ weights_in_filter ][ k ]
                    weights_avg += self.Convolution_layer1['weights'][ input_layer_node ][ Processing_Node ]
                weights_avg = round( weights_avg / len( self.Convolution_layer1['init'] ), 8 )
                # load each weight
                for weights_in_filter in range( len( self.Convolution_layer1['init'] ) ): # 24
                    input_layer_node = self.Convolution_layer1['init'][ weights_in_filter ][ k ]
                    self.Convolution_layer1['weights'][ input_layer_node ][ Processing_Node ] = weights_avg



    def sigmoid( self, x, derivative = False ):
        if derivative:
            sigmoid = 1 / ( 1 + np.exp( -x ) )
            return sigmoid * ( 1 - sigmoid )
        # -x because number in negative has the range from 0 to 1
        # np.exp() + 1 because we have the 1 / 1 + np.exp( -x )
        # if you do not plus one, the number under 1 will get a
        # very big number
        return 1 / ( 1 + np.exp( -x ) )

    def softmax( self, x, derivative = False ):
        exps = np.exp( x - np.max( x ) )
        cal = exps / np.sum( exps )
        if derivative:
            return cal * ( 1 - cal )
        else:
            return cal

    def relu( self, x, derivative = False ):
        if derivative:
            x[ x <= 0 ] = 0
            x[ x > 0 ] = 1
            return x
        return np.maximum( x, 0 )


    def GradientDescentOptimizer( self, LearningRate = 0.01 ):
        layer2_Error = self.labels - self.layer2['prediction']
        self.layer2['delta'] = layer2_Error * self.relu( self.layer2['prediction'], derivative = True )
        layer1_Error = self.layer2['delta'].dot( self.layer2['weights'].T )
        self.layer1['delta'] = layer1_Error * self.softmax( self.layer1['prediction'], derivative = True )
        Convolution_layer1_Error = self.layer1['delta'].dot( self.layer1['weights'].T )
        self.Convolution_layer1['delta'] = Convolution_layer1_Error * self.Convolution_layer1['output']
        
        # update weights
        self.layer2['weights'] += self.layer1['prediction'].T.dot( self.layer2['delta'] ) * LearningRate
        self.layer1['weights'] += self.Convolution_layer1['output'].T.dot( self.layer1['delta'] ) * LearningRate
        self.Convolution_layer1['weights'] += self.input_layer.T.dot( self.Convolution_layer1['delta'] ) * LearningRate

        # do share weights
        self.Convolution_Reconstruct()
        return np.mean( np.abs( layer2_Error ) )

    def AdamOptimizer( self, LearningRate = 0.01 ):
        pass


    def train( self, dataset, num_of_training = 40000, print_process = True ):

        if print_process:
            print("\033[2;2H{}".format('------------- Training Rate -------------'))
            print("\033[3;2H{}".format('[                              ]        %'))

        Error = 0

        for loop in range( num_of_training ):
            # for training images
            self.input_layer = np.expand_dims( dataset.images[ loop ], axis = 0 )
            # for training labels
            self.labels = np.expand_dims( dataset.labels[ loop ], axis = 0 )

            # forward
            self.Convolution_layer1['output'] = np.dot( self.input_layer, self.Convolution_layer1['weights'] )
            self.layer1['prediction'] = self.relu( np.dot( self.Convolution_layer1['output'], self.layer1['weights'] ) )
            self.layer2['prediction'] = self.softmax( np.dot( self.layer1['prediction'], self.layer2['weights'] ) )

            Error += self.GradientDescentOptimizer()
            if print_process:
                self.PRINT_RATE( num_of_training, loop )

        print( 'Loss in epoches: {}'.format( Error ) )


    def test( self, dataset, num_of_testing = 1000 ):
        correct = 0

        for loop in range( num_of_testing ):
            # for testing images
            self.input_layer = np.expand_dims( dataset.images[ loop ], axis = 0 )
            # for testing labels
            self.labels = np.expand_dims( dataset.labels[ loop ], axis = 0 )

            # forward
            self.Convolution_layer1['output'] = np.dot( self.input_layer, self.Convolution_layer1['weights'] )
            self.layer1['prediction'] = self.relu( np.dot( self.Convolution_layer1['output'], self.layer1['weights'] ) )
            self.layer2['prediction'] = self.softmax( np.dot( self.layer1['prediction'], self.layer2['weights'] ) )

            # check the answer is true
            if np.argmax( self.layer2['prediction'] ) == np.argmax( self.labels ):
                correct += 1
        return round( correct / num_of_testing, 2 )

    def Save( self, file_name = 'CNN_master.npz' ):
        np.savez( file_name, 
            layer1 = self.layer1['weights'], 
            layer2 = self.layer2['weights'], 
            Convolution_layer1 = self.Convolution_layer1['weights'] )
    def Restore( self, file_name = 'CNN_master.npz' ):
        data = np.load( file_name )
        self.layer1['weights'] = data['layer1']
        self.layer2['weights'] = data['layer2']
        self.Convolution_layer1['weights'] = data['Convolution_layer1']

    def PRINT_RATE( self, i, loop ):
        num = int( i / 30 )
        count = math.ceil( loop / num )
        count = count if count <= 31 else 31
        if loop % num == 0:
            print( "\033[3;3H{}".format( '#'*count ) )
        print( "\033[3;36H{}".format( round( ( loop / i )*100 + 0.01, 2 ) ) )

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets( "MNIST_data/", one_hot = True )

    epochs = 50

    Network = Model()

    for i in range( epochs ):
        if i == 0:
            os.system('clear')
        print( "\033[1;2HEpoch {}".format( i + 1 ) )
        Network.train( mnist.train, print_process = False )
        print( 'Training set Correct: {}'.format( Network.test( mnist.train ) ) )
        print( 'Testing set Correct: {}'.format( Network.test( mnist.test ) ) )
        Network.Save()
