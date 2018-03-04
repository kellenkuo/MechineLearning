import numpy as np
import math
import os

''' input 784 -> Convolution1 -> Convolution2 -> HiddenLayer 100 -> output 10,
    without max pooling
'''

class Model( object ):

    input_layer = None
    labels = None

    kernel = { 'Convolution_layer1': None, 'Convolution_layer2': None }
    layer1 = { 'weights': None, 'prediction': None, 'delta': None  }
    layer2 = { 'weights': None, 'prediction': None, 'delta': None }

    def __init__( self ):
        self.input_layer = np.random.random(( 1, 64*28*28 ))
        self.labels = np.random.random(( 1, 10 ))
        self.layer1['weights'] = 2 * np.random.random(( 64*28*28, 2000 )) - 1
        self.layer1['prediction'] = np.random.random(( 1, 2000 ))
        self.layer2['weights'] = 2 * np.random.random(( 2000, 10 )) - 1
        self.layer2['prediction'] = np.random.random(( 1, 10 ))
        # Convolution
        self.kernel['Convolution_layer1'] = np.random.random(( 16, 5, 5 ))
        self.kernel['Convolution_layer2'] = np.random.random(( 4, 5, 5 ))

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

    def Convolution( self, data ):
        # import scipy library
        from scipy import signal
        # num_of kernel equal 16
        num_of_kernel_1 = self.kernel['Convolution_layer1'].shape[0]
        # to put the final result
        layer1 = np.zeros(( num_of_kernel_1, 28, 28 ))
        # reshape ( 1, 784 ) to ( 28, 28 )
        train_image = np.reshape( data, ( 28, 28 ) )
        # layer1 convolution
        for i in range( num_of_kernel_1 ):
            tmp[i] = signal.convolve2d( train_image, self.kernel['Convolution_layer1'][ i ], boundary = 'symm', mode = 'same' )
        # for kernel two same as kernel one
        num_of_kernel_2 = self.kernel['Convolution_layer2'].shape[0]
        # Output is 16 * 4 image so we create ( 4, 18, 28, 28 ) matrix
        layer2 = np.zeros(( num_of_kernel_2, num_of_kernel_1, 28, 28 ))
        for i in range( num_of_kernel_2 ):
            for k in range( num_of_kernel_1 ):
                layer2[i][k] = signal.convolve2d( layer1[ num_of_kernel_1 ], self.kernel['Convolution_layer2'][ num_of_kernel_2 ], boundary = 'symm', mode = 'same' )
        return layer2.flatten()

    def Convolution_old( self, data ):
        from scipy import signal
        Filter = np.array([[ 0, -1, 0 ],
                           [-1, 5, -1 ],
                           [ 0, -1, 0 ]])
        # 1*784 -> 28*28
        train_image = np.reshape( data, ( 28, 28 ) )
        # 28*28 -> 26*26
        train_image = signal.convolve2d( train_image, Filter, boundary = 'symm', mode = 'same' )
        # 26*26 -> 1*676
        train_image = train_image.flatten()
        return train_image


    def GradientDescentOptimizer( self, LearningRate = 0.01 ):
        layer2_Error = self.labels - self.layer2['prediction']
        self.layer2['delta'] = layer2_Error * self.relu( self.layer2['prediction'], derivative = True )
        layer1_Error = self.layer2['delta'].dot( self.layer2['weights'].T )
        self.layer1['delta'] = layer1_Error * self.softmax( self.layer1['prediction'], derivative = True )
        
        self.layer2['weights'] += self.layer1['prediction'].T.dot( self.layer2['delta'] ) * LearningRate
        self.layer1['weights'] += self.input_layer.T.dot( self.layer1['delta'] ) * LearningRate
        return np.mean( np.abs( layer2_Error ) )

    def AdamOptimizer( self, LearningRate = 0.01 ):
        pass


    def train( self, dataset ):

        print("\033[2;2H{}".format('------------- Training Rate -------------'))
        print("\033[3;2H{}".format('[                               ]       %'))

        for loop in range(40000):
            convolution_image = self.Convolution( dataset.images[ loop ] )
            # for training images
            self.input_layer = np.expand_dims( convolution_image, axis = 0 )
            # for training labels
            self.labels = np.expand_dims( dataset.labels[ loop ], axis = 0 )

            # forward
            self.layer1['prediction'] = self.relu( np.dot( self.input_layer, self.layer1['weights'] ) )
            self.layer2['prediction'] = self.softmax( np.dot( self.layer1['prediction'], self.layer2['weights'] ) )

            Error = self.GradientDescentOptimizer()
            self.PRINT_RATE( 40000, loop )


    def test( self, dataset, num_of_testing = 1000 ):
        correct = 0

        for loop in range( num_of_testing ):
            # for testing images
            self.input_layer = np.expand_dims( dataset.images[ loop ], axis = 0 )
            # for testing labels
            self.labels = np.expand_dims( dataset.labels[ loop ], axis = 0 )

            # forward
            self.layer1['prediction'] = self.relu( np.dot( self.input_layer, self.layer1['weights'] ) )
            self.layer2['prediction'] = self.softmax( np.dot( self.layer1['prediction'], self.layer2['weights'] ) )

            # check the answer is true
            if np.argmax( self.layer2['prediction'] ) == np.argmax( self.labels ):
                correct += 1
        return round( correct / num_of_testing, 2 )

    def Save( self, file_name = 'weights.npz' ):
        np.savez( file_name, 
            layer1 = self.layer1['weights'], 
            layer2 = self.layer2['weights'], 
            Convolution_layer1 = self.kernel['Convolution_layer1'], 
            Convolution_layer2 = self.kernel['Convolution_layer2'] )
    def Restore( self, file_name = 'weights.npz' ):
        data = np.load( file_name )
        self.layer1['weights'] = data['layer1']
        self.layer2['weights'] = data['layer2']
        self.kernel['Convolution_layer1'] = data['Convolution_layer1']
        self.kernel['Convolution_layer2'] = data['Convolution_layer2']

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

    Network = Model()

    for i in range(1):
        if i == 0:
            os.system('clear')
        Network.train( mnist.train )
    Model.Save('CNN_Network.npz')
