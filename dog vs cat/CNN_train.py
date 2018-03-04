from GrabCatDog import data
import tensorflow as tf
from os import system
import math

""" training data """
CatDog = data()

''' set GPU Memory '''
gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.9 )

epochs = 1
batch_size = 10

layer1_nodes = 500

classifier = 2

def Converlution1( image ):
    x_image = tf.reshape( image, [ -1, 144, 176, 1 ] )
    weights_converlution = tf.Variable( tf.truncated_normal( [ 5, 5, 1, 32 ], stddev = 0.1 ) )
    converlution = tf.nn.relu( tf.nn.conv2d( x_image, weights_converlution, strides = [ 1, 1, 1, 1 ], padding = 'SAME' ) )
    max_pooling = tf.nn.max_pool( converlution, ksize = [ 1, 2, 2, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
    return max_pooling
def Converlution2( image ):
    x_image = Converlution1( image )
    weights_converlution = tf.Variable( tf.truncated_normal( [ 5, 5, 32, 64 ], stddev = 0.1 ) )
    converlution = tf.nn.relu( tf.nn.conv2d( x_image, weights_converlution, strides = [ 1, 1, 1, 1 ], padding = 'SAME' ) )
    max_pooling  = tf.nn.max_pool( converlution, ksize = [ 1, 2, 2, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
    return tf.reshape( max_pooling, [ -1, 36*44*64 ] )
def Converlution3( image ):
    x_image, _ = Converlution2( image )
    weights_converlution = tf.Variable( tf.truncated_normal( [ 5, 5, 64, 128 ], stddev = 0.1 ) )
    converlution = tf.nn.relu( tf.nn.conv2d( x_image, weights_converlution, strides = [ 1, 1, 1, 1 ], padding = 'SAME' ) )
    max_pooling  = tf.nn.max_pool( converlution, ksize = [ 1, 2, 2, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
    return max_pooling, tf.reshape( max_pooling, [ -1, 18*22*128 ] )
def Converlution4( image ):
    x_image, _ = Converlution3( image )
    weights_converlution = tf.Variable( tf.truncated_normal( [ 5, 5, 128, 256 ], stddev = 0.1 ) )
    converlution = tf.nn.relu( tf.nn.conv2d( x_image, weights_converlution, strides = [ 1, 1, 1, 1 ], padding = 'SAME' ) )
    max_pooling  = tf.nn.max_pool( converlution, ksize = [ 1, 2, 2, 1 ], strides = [ 1, 2, 2, 1 ], padding = 'SAME' )
    return tf.reshape( max_pooling, [ -1, 9*11*256 ] )

def Neural_Network_Model( input_data ):
    HiddenLayer_1 = { 'weights': tf.Variable( tf.random_normal( [ 36*44*64, layer1_nodes ] ) ), 'biase': tf.Variable( tf.zeros( [layer1_nodes ] ) ) }
    OutputLayer = { 'weights': tf.Variable( tf.random_normal( [ layer1_nodes, classifier ] ) ) }

    Layer1 = tf.matmul( input_data, HiddenLayer_1['weights'] ) + HiddenLayer_1['biase']
    Layer1 = tf.nn.relu( Layer1 )
    Output = tf.matmul( Layer1, OutputLayer['weights'] )

    return Output

def PRINT_RATE( i, loop ):
    num = int( i / 30 )
    count = math.ceil( loop / num )
    count = count if count <= 31 else 31
    if loop % num == 0:
        print( "\033[3;3H{}".format( '#'*count ) )
    print( "\033[3;36H{}".format( round( ( loop / i )*100 + 0.03, 2 ) ) )

def Main(  ):
    Input = tf.placeholder( tf.float32, shape = [ None, 25344 ] )
    Ans = tf.placeholder( tf.float32, shape = [ None, 2 ] )

    prediction = Neural_Network_Model( Converlution2( Input ) )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = Ans, logits = prediction ) )
    optimizer = tf.train.AdamOptimizer(  ).minimize( cost )
#    optimizer = tf.train.AdadeltaOptimizer( 0.1 ).minimize( cost )

    Saver = tf.train.Saver()

    with tf.Session( config = tf.ConfigProto( gpu_options = gpu_options ) ) as sess:
        sess.run( tf.global_variables_initializer() )
#        Saver.restore( sess, "save/CNN_NET.ckpt" )

        system('clear')
        print("\033[2;2H{}".format('------------- Learning Rate ------------'))
        print("\033[3;2H{}".format('[                               ]      %'))

        count = 0

        for loop in range( epochs ):
            epoch_loss = 0
            num_of_dataset = int( CatDog.num_of_train_example() / batch_size )
            for i in range( num_of_dataset ):
                count += 1
                PRINT_RATE( num_of_dataset * epochs, count )
                train_Input, train_Ans = CatDog.train_next_batch( batch = batch_size )
                _, c = sess.run( [ optimizer, cost ], feed_dict = { Input: train_Input, Ans: train_Ans } )
                epoch_loss += c
            print("\033[4;1H Loss: {}".format( epoch_loss ))

            correct = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( Ans, 1 ) )
            accuracy = tf.reduce_mean( tf.cast( correct, tf.float32 ) )
            test_Input, test_Ans = CatDog.test( 100 )
            print("\033[5;1H epoch{} prediction: {}".format( loop, sess.run( accuracy, feed_dict = { Input: test_Input, Ans: test_Ans } )*100 ) )

            save_path = Saver.save( sess, "save/CNN_NET.ckpt" )
            print(' Successful: epoch', str( loop ), ' Save to file -> ', save_path )

if __name__ == "__main__":
    Main(  )
