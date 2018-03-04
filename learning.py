import numpy as np

''' input 784 -> HiddenLayer 100 -> output 10 '''

class Network( object ):

    input_layer = 0
    labels = 0
    layer1_weight = 2 * np.random.random(( 2, 5 )) - 1
    layer1 = 0
    output_weights = 2 * np.random.random(( 5, 1 )) - 1
    output = 0

    layer1 = { 'weights': 2 * np.random.random(( 784, 100 )) - 1,
                'prediction': 0 }
    layer2 = { 'weights': 2 * np.random.random(( 100, 10 )) - 1 ,
                'prediction': 0 }

    def init( self ):
        pass

    def sigmoid( self, x, derivative = False ):
        if derivative:
            return x * ( 1 - x )
        # -x because number in negative has the range from 0 to 1
        # np.exp() + 1 because we have the 1 / 1 + np.exp( -x )
        # if you do not plus one, the number under 1 will get a
        # very big number
        return 1 / ( 1 + np.exp( -x ) )
    '''
    def Backpropagation( self, layer ):
        error = self.labels - layer
        delta = error * self.sigmoid( layer, True )
        self.layer1_weight += np.dot( self.input_layer.T, delta )
        return np.sum( error )
    '''
    def Backpropagation():
        layer2_Error = self.labels - Layer2_prediction
        layer2_delta = layer2_Error * self.sigmoid( Layer2_prediction, derivative = True )
        layer1_Error = layer2_delta.dot( self.layer2.T )
        layer1_delta = layer1_Error * self.sigmoid( Layer1_prediction, derivative = True )

        self.layer2['weights'] += Layer2_prediction.T.dot( layer2_delta )
        self.layer1['weights'] += Layer1_prediction.T.dot( layer1_delta )

    # the input must be numpy array
    def train( self, data, label ):
        self.input_layer = data
        self.labels = label.T

        for loop in range(60000):
            self.layer1 = np.dot( self.input_layer, self.layer1_weight )
            self.layer1 = self.sigmoid( self.layer1 )
            Error = self.Backpropagation( self.layer1 )

            if ( loop + 1 ) % 100 == 0:
                print( Error )

    def train( self, dataset ):
        # for training images
        self.input_layer = dataset[0]
        # for training Answer
        self.labels = dataset[1]

        for loop in range(60000):



if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets( "MNIST_data/", one_hot = True )

    data = np.array([[1,0]])
    label = np.array([[0]])

    ML = Network()
    ML.train( data, label )
