from GrabMNIST import data
import numpy as np
import csv
import math
import Method

precision = 6
method = 'softmax'

file_name = 'Weight_Save_CNN_weight1_' + method + '.csv'

call = data()
test_image, test_ans = call.GrabTest(  )

# 9 picture Input
Input = np.zeros( (756), dtype = np.int16 )
# 9 picture weight
weight1 = np.zeros( (756,10), dtype = np.float64 )
# y = a*x + bias
bias = np.zeros( (10), dtype = np.float64 )
# create output
layer = np.zeros( (10), dtype = np.float64 )
# create weight2
# weight2 = np.zeros( (10), dtype = np.float64 )
# create output
output = np.zeros( (10), dtype = np.float64 )
print("Create variable ----- Done")

weight1 = np.genfromtxt( file_name, delimiter = ',' )
# bias = np.genfromtxt( "bias.csv", delimiter = ',' )
# weight2 = np.genfromtxt( "Weight_Save_CNN_weight2.csv", delimiter = ',' )
print("Load weight ----- Done")


def FEED_TRAIN_DATA( loop ):
    global Input
    for i in range(756):
        total = 0
        for j in range(9):
            total += int( test_image[loop][ i + j ] )
        Input[i] = round( total / 9 )

''' network calculate '''
def LAYER_CALCULATE():
    global layer
    for i in range(10):
        total = 0
        for j in range(756):
            total += Input[ j ] * weight1[ j, i ]
        layer[ i ] = round( total + bias[ i ], precision )
def OUTPUT_CALCULATE():
    global output
    if method == 'sigmoid':
        for i in range(10):
            output[ i ] = round( Method.sigmoid( layer[ i ] ), precision )
    elif method == 'softmax':
        output = Method.softmax( layer )

def CORRECT_RATE( loop ):
    if int( test_ans[ loop ][0] ) == np.argmax( output ):
        return True
    else:
        return False

''' extend function '''
def PRINT_RATE( i, loop ):
    num = int( i / 30 )
    count = math.ceil( loop / num )
    count = count if count <= 31 else 31
    if loop % num == 0:
        print( "\033[6;3H{}".format( '#'*count ) )
    print( "\033[6;36H{}%".format( round( ( loop / i )*100 + 0.05, 2 ) ) )

def Main( i ):
    print("\033[5;2H{}".format('------------- Testing Rate -------------'))
    print("\033[6;2H{}".format('[                               ]'))

    tmp = 0
    for loop in range(i):       
        PRINT_RATE( i, loop )
        FEED_TRAIN_DATA( loop )
        LAYER_CALCULATE( )
        OUTPUT_CALCULATE( )
        if CORRECT_RATE( loop ) == True:
            tmp += 1
    print( "Correct RATE: {}%".format( round( ( tmp / i  ) * 100, 2 ) ) )


if __name__ == "__main__":
    print( 'Test DataSet Sample -> ' + str( len( test_ans ) ) )
    Main( 100 )

