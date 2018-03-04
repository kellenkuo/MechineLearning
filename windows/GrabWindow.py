from skimage import io, color, filters, img_as_float
from skimage.transform import rescale, resize
import numpy as np
import random
import os

# image shape 1920*1080
# one_hot label
# window_one = [ 1, 0, 0, 0 ]

# filters.gaussian or filters.hessian

class data( object ):
    count = []

    def __init__( self ):
        for i in range(67):
            self.count.append( False )

    def train( self, image_size = 1920*1080 ):
        train_image = np.zeros( ( 67, image_size ), dtype = 'float32' )
        train_label = np.zeros( ( 67, 6 ), dtype = 'uint8' )

        tmp = 0
        while False in self.count:
            rand_number = random.randint( 0, 66 )
            if self.count[ rand_number ] == False:
                if rand_number < 8:
                    image_number = str( 9213 + rand_number )
                    folder = 1
                elif rand_number < 21:
                    image_number = str( 9227 + rand_number )
                    folder = 2
                elif rand_number < 30:
                    image_number = str( 9235 + rand_number )
                    folder = 3
                elif rand_number < 44:
                    image_number = str( 9241 + rand_number )
                    folder = 4
                elif rand_number < 60:
                    image_number = str( 9249 + rand_number )
                    folder = 5
                elif rand_number < 67:
                    image_number = str( 9258 + rand_number )
                    folder = 6

                image = color.rgb2gray( io.imread( os.path.join("train/one/" + str( folder ) + "/" + str( image_number ) + ".jpg") ) )
                image = filters.gaussian( image )
                image = np.reshape( img_as_float( resize( image, ( 1080, 1920 ), mode = 'reflect' ) ).astype( np.float32 ), -1 )
                train_image[ tmp ] = image

                for i in range(6):
                    if i == folder:
                        train_label[ tmp ][i] = 1
                    else:
                        train_label[ tmp ][i] = 0

                tmp += 1
                self.count[ rand_number ] = True
        return train_image, train_label

