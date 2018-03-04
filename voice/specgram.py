import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import sys

def convert( voice_path ):
    sample_rate, x = wavfile.read( voice_path )
    
    fig = plt.figure()
    plt.specgram( x, Fs = sample_rate, xextent = ( 0, 1 ) )
    plt.xticks([])
    plt.yticks([])
    plt.margins( 0, 0 )
    fig.canvas.draw()
    data = np.fromstring( fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '' )
    data = data.reshape( fig.canvas.get_width_height()[::-1] + ( 3, ) )
    # useful data [58:428][80:577][:]
#    fig.savefig( '0.png', bbox_inches='tight', pad_inches = 0 )
    plt.close( fig )
    return useful_area( data )

def useful_area( np_array ):
    select = np.zeros( ( 400, 500, 3 ), dtype = 'float32' )
    white = np.array([ 1., 1., 1. ])
    for i in range( 40, 440 ):
        for k in range( 78, 578 ):
            select[ i - 40 ][ k - 78 ] = np_array[ i ][ k ] / 255
            check = select[ i - 40 ][ k - 78 ] == white
            if not False in check:
                select[ i - 40 ][ k - 78 ] = np.array([ 0., 0., 0. ])
    return np.round( select, 8 )

if __name__ == '__main__':
    if len( sys.argv ) > 1:
        path = sys.argv[1]
        print( convert( path ).shape )
        print( 'Successful convert {} to 0.png'.format( path ) )
    else:
        sys.exit()
