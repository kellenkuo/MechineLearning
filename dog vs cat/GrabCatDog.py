from skimage import io, color, img_as_float
from skimage.transform import rescale, resize
import numpy as np
import random
import os

# image shape [ 414, 500, 3 ]
# [ 0, 1 ] == dog
# [ 1, 0 ] == cat

class data( object ):
    count = []

    def __init__( self ):
        for i in range(24000):
            self.count.append( False )

    def train( self ):
        pass

    def num_of_train_example( self ):
        return 24000

    def num_of_test_example( self ):
        return 1000
    '''
    def train_next_batch( self, batch = 500 ):
        train_image = []
        train_label = []
        tmp = 0

        # check if has been load training data is full
        if not False in self.count:
            for i in range(12000):
                self.count[i] = False

        while tmp < batch:
            rand_number = random.randint( 0, 11999 )
            if self.count[ rand_number ] == False:
                train_image.append( np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/cat." + str( rand_number ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 ) )
                train_image.append( np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/dog." + str( rand_number ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 ) )
                train_label.append( [ 1, 0 ] )
                train_label.append( [ 0, 1 ] )
                self.count[ rand_number ] = True
                tmp += 1
        return train_image, train_label
    '''
    def train_next_batch( self, batch = 100 ):
        train_image = np.zeros( ( batch, 25344 ), dtype = 'float32' )
        train_label = np.zeros( ( batch, 2 ), dtype = 'uint8' )
        tmp = 0

        # check if has been load training data is full
        if not False in self.count:
            for i in range(24000):
                self.count[i] = False

        while tmp < batch:
            rand_number = random.randint( 0, 23999 )
            if self.count[ rand_number ] == False:
                if rand_number < 12000:
                    image = np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/cat." + str( rand_number ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 )
                    train_image[ tmp ] = image
                    train_label[ tmp ][0] = 1
                    train_label[ tmp ][1] = 0
                if rand_number >= 12000:
                    image = np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/dog." + str( rand_number - 12000 ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 )
                    train_image[ tmp ] = image
                    train_label[ tmp ][0] = 0
                    train_label[ tmp ][1] = 1
                
                self.count[ rand_number ] = True
                tmp += 1
        return train_image, train_label

    def test( self, cache = 100 ):
        test_image = np.zeros( ( cache, 25344 ), dtype = 'float32' )
        test_label = np.zeros( ( cache, 2 ), dtype = 'uint8' )
        test_count = []
        tmp = 0

        while tmp < cache if cache <= 500 else 500:
            rand_number = random.randint( 0, 499 )
            if not rand_number in test_count:
                image = np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/cat." + str( rand_number + 12000 ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 )
                test_image[ tmp ] = image
                test_label[ tmp ][0] = 1
                test_label[ tmp ][1] = 0
            
                image = np.reshape( img_as_float( resize( color.rgb2gray( io.imread( os.path.join("train/dog." + str( rand_number + 12000 ) + ".jpg") ) ), ( 144, 176 ), mode = 'reflect' ) ).astype( np.float32 ), -1 )
                test_image[ tmp + 1 ] = image
                test_label[ tmp + 1 ][0] = 0
                test_label[ tmp + 1 ][1] = 1

                test_count.append( rand_number )
                tmp += 2
        return test_image, test_label
