from specgram import convert
import numpy as np
import random
import os

class data( object ):
    count = []
    num_of_dataset = 0
    DATA_Category = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow" ]
    num_of_each_category = []
    DATA_DIR = 'string'

    def __init__( self ):
        tmp = 0
        for i in range(20):
            self.DATA_DIR = 'train/' + self.DATA_Category[i]
            How_many_file = self.Search_for_all( self.DATA_DIR )
            self.num_of_each_category.append( How_many_file )
            self.num_of_dataset += How_many_file
            print( 'Finish Reading Dataset -> {}'.format( self.DATA_Category[i] ) )
        # init count by num_of_dataset * False
        for i in range( self.num_of_dataset ):
            self.count.append( False )

    def Search_for_all( self, DIR ):
        file_data = []
        for filename in os.listdir( DIR ):
            file_data.append( filename )
        return len( file_data )

    def num_in_dataset( self ):
        return self.num_of_dataset

    def train_next_batch( self, batch_size = 500 ):
        train_image = np.zeros( ( batch_size, 400, 500, 3 ), dtype = 'float32' )
        train_label = np.zeros( ( batch_size, 20 ), dtype = 'uint8' )

        if not False in self.count:
            for i in range( self.num_of_dataset ):
                self.count[i] = False

        tmp = 0
        # init screen
        os.system('clear')
        while tmp < batch_size:
            # Grab random number
            while( True ):
                random_number = random.randint( 1, self.num_of_dataset )
                if self.count[ random_number ] == False:
                    self.count[ random_number ] = True
                    break
            # Grab image from specgram
            path, label = self.Convert_DATA_DIR( random_number )
            print("\033[1;1H{}".format( ' ' * 60 ))
            print("\033[1;1HReading Voice {}: {}".format( tmp, path ))
            train_image[ tmp ] = convert( path )
            train_label[ tmp ][ label ] = 1
            tmp += 1
        return train_image, train_label


    def Convert_DATA_DIR( self, number ):
        find_target = number
        category = 0
        while( True ):
            if find_target < self.num_of_each_category[ category ]:
                break
            else:
                find_target -= self.num_of_each_category[ category ]
                category += 1

        filename = os.listdir( 'train/' + self.DATA_Category[ category ] )[ find_target ]

        return 'train/' + self.DATA_Category[ category ] + '/' + filename, category
