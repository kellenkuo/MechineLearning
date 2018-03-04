from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( "MNIST_data/", one_hot = False )

batch_size = 100
epochs = 20

input_nodes = 784
encoder_nodes_1 = 300
encoder_nodes_2 = 100
code_nodes = 10
decoder_nodes_2 = 100
decoder_nodes_1 = 300
output_nodes = 784

def PRINT_RATE( i, loop ):
    num = int( i / 30 )
    count = math.ceil( loop / num )
    count = count if count <= 31 else 31
    if loop % num == 0:
        print( "\033[3;3H{}".format( '#'*count ) )
    print( "\033[3;36H{}%".format( round( ( loop / i )*100 + 0.05, 2 ) ) )

def weight_variable( shape, name ):
    return tf.Variable( tf.random_normal( shape = shape, stddev = 0.1 ), name )
def bias_variable( shape, name ):
    return tf.Variable( tf.constant( 0.1, shape = shape ), name )

weights = {
    'encoder_1': weight_variable( [ input_nodes, encoder_nodes_1 ], 'encoder_1_weights' ),
    'encoder_2': weight_variable( [ encoder_nodes_1, encoder_nodes_2 ], 'encoder_2_weights' ),
    'code': weight_variable( [ encoder_nodes_2, code_nodes ], 'code_weights' )
}

biases = {
    'encoder_1': bias_variable( [ encoder_nodes_1 ], 'encoder_1_biases' ),
    'encoder_2': bias_variable( [ encoder_nodes_2 ], 'encoder_2_biases' ),
    'code': bias_variable( [ code_nodes ], 'code_biases' )
}

def auto_encoder( x ):
    encoder_layer_1 = tf.nn.sigmoid( tf.add( tf.matmul( x, weights['encoder_1'] ), biases['encoder_1'] ) )
    encoder_layer_2 = tf.nn.sigmoid( tf.add( tf.matmul( encoder_layer_1, weights['encoder_2'] ), biases['encoder_2'] ) )
    code_layer      = tf.nn.sigmoid( tf.add( tf.matmul( encoder_layer_2, weights['code'] ), biases['code'] ) )

    return code_layer

def second_network( x ):
    with tf.variable_scope('final'):
        output = {
            'weights': weight_variable( [ 10, 10 ], 'final_weight' ),
            'biases': bias_variable( [ 10 ], 'finale_bias' )
        }
    return tf.add( tf.matmul( x, output['weights'] ), output['biases'] )

'''
save_var = {
    'encoder_1_weights': weights['encoder_1'],
    'encoder_2_weights': weights['encoder_2'],
    'code_weights': weights['code'],
    'encoder_1_biases': biases['encoder_1'],
    'encoder_2_biases': biases['encoder_2'],
    'code_biases': biases['code']
}
'''

def Main():
    Input = tf.placeholder( tf.float32, shape = [ None, 784 ] )
    Ans = tf.placeholder( tf.uint8, shape = [ None, 10 ] )

    # throw dataset into auto_encoder netowrk
    auto_encoder_output = auto_encoder( Input )
    prediction = second_network( auto_encoder_output )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = Ans, logits = prediction ) )
    optimizer = tf.train.AdamOptimizer().minimize( cost, var_list = [ tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = 'final') ] )

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        Saver = tf.train.Saver()
        Saver.restore( sess, "save/CNN_NET.ckpt" )

        os.system('clear')

        print("\033[2;2H{}".format('------------ Learning Rate ------------'))
        print("\033[3;2H{}".format('[                               ]'))

        count = 0

        for loop in range( epochs ):
            epoch_loss = 0
            num_of_dataset = int( mnist.train.num_examples / batch_size )
            for i in range( num_of_dataset ):
                count += 1
                PRINT_RATE( num_of_dataset * epochs, count )
                train_Input, train_Ans = mnist.train.next_batch( batch_size )
                _, c = sess.run( [ optimizer, cost ], feed_dict = { Input: train_Input, Ans: train_Ans } )
                epoch_loss += c
            print("\033[4;1H Loss: {}".format( epoch_loss ))

            correct = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( Ans, 1 ) )
            accuracy = tf.reduce_mean( tf.cast( correct, tf.float32 ) )
            print("\033[5;1H Accuracy: {}".format( accuracy.eval( { Input: mnist.test.images, Ans: mnist.test.labels } ) ))


        '''
        result = result * 255
        result = result.astype( np.uint8 )
        pixels = result.reshape(( 28, 28 ))
        plt.imshow( pixels, cmap = 'gray' )
        plt.show()
        '''

if __name__ == "__main__":
    Main()

