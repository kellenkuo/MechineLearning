from specgram_for_image import convert
import multiprocessing as mp
from scipy.misc import imsave
import numpy as np
import os

DATA_Category = [ "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow" ]
def job( q ):
    for i in range(30):
        for filename in os.listdir( 'train/' + DATA_Category[i] ):
            image = np.zeros( ( 400, 500, 3 ), dtype = 'uint8' )
            image = convert( 'train/' + DATA_Category[i] + '/' + filename )
            imsave( 'train_by_image/' + DATA_Category[i] + '/' + filename[:-4] + '.jpg', image )
            print( 'Successful Save {} to image'.format( 'train/' + DATA_Category[i] + '/' + filename ) )
    print('Finish')

if __name__ == "__main__":
    q = mp.Queue()
    process_one = mp.Process( target = job, args = ( q, ) )
    process_one.start()
    process_one.join()