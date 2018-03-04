from GrabMNIST import data
import numpy as np
import csv
import math
import random
import Method

precision = 6
Learning_Rate_1 = 0.0001
# Learning_Rate_2 = 0.00001
Load_Weight = False

def weight_init_variable( method ):
	if method == 'sigmoid':
		return random.uniform( 1.0, 1.1 )
	elif method == 'softmax':
		return 0.0001

call = data()
train_image, train_ans = call.GrabTrain(  )

# 9 picture Input
Input = np.zeros( (756), dtype = np.float16 )
# 9 picture weight
weight1 = np.zeros( (756,800), dtype = np.float64 )
# create layer1
layer1 = np.zeros( (800), dtype = np.float64 )
# create output
output1 = np.zeros( (800), dtype = np.float16 )
# create HiddenError
HiddenError = np.zeros( (800), dtype = np.float16 )
# create weight2
weight2 = np.zeros( (800,10), dtype = np.float64 )
# create layer2
layer2 = np.zeros( (10), dtype = np.float16 )
# create error
error = np.zeros( (10), dtype = np.float16 )
# create output
output2 = np.zeros( (10), dtype = np.float32 )
print("Create variable ----- Done")

def FEED_TRAIN_DATA( loop ):
	global Input
	for i in range(756):
		total = 0
		for j in range(9):
			total += int( train_image[loop][ i + j ] )
		Input[i] = round( total / 9 )


''' network calculate '''
def LAYER1_CALCULATE():
	global layer
	for i in range(800):
		total = 0
		for j in range(756):
			total += Input[ j ] * weight1[ j, i ]
		layer1[ i ] = round( total , precision )
	layer1_max = max( layer1 )
	for i in range(800):
		layer1[ i ] = round( layer1[ i ] / layer1_max , precision )
def OUTPUT1_CALCULATE():
	global output1
	for i in range(800):
		output1[ i ] = round( Method.sigmoid( layer1[ i ] ), precision )
def LAYER2_CALCULATE():
	global layer
	for i in range(10):
		total = 0
		for j in range(800):
			total += output1[ j ] * weight2[ j, i ]
		layer2[ i ] = round( total , precision )
def OUTPUT2_CALCULATE(  ):
	global output2
	output2 = Method.softmax( layer2 )
def TOTAL_FORWARD():
	LAYER1_CALCULATE()
	OUTPUT1_CALCULATE()
	LAYER2_CALCULATE()
	OUTPUT2_CALCULATE()



def ERROR_CALCULATE( ans ):
	global error
	for i in range(10):
		if i == ans:
			error[i] = 1 - output2[i]
		else:
			error[i] = 0 - output2[i]
def HIDDENERROR_CALCULATE( ans ):
	for i in range(800):
		for j in range(10):
			HiddenError[ i ] += round( ( error[ j ] / weight2[ i, j ] ), precision )
def WEIGHT1_REVISE(  ):
	global weight1
	for i in range(800):
		for j in range(756):
			weight1[ j, i ] += round( Input[ j ] * HiddenError[ i ] * Method.sigmoid_reverse( layer1[ i ] ) * 0.00001 * Learning_Rate_1, precision )
def WEIGHT2_REVISE(  ):
	for i in range(10):
		for j in range(800):
			weight2[ j, i ] += round( output1[ j ] * error[ i ] * 0.0001 * Learning_Rate_1, precision )	
def TOTAL_REIVSE( loop ):
	for i in train_ans[ loop ]:
		ans = int( i )
	ERROR_CALCULATE( ans )
	WEIGHT2_REVISE(  )
	HIDDENERROR_CALCULATE( ans )
	WEIGHT1_REVISE(  )


def global_initialization_variable():
	global weight1, weight2
	# weight init to 0.001 before load data
	if Load_Weight:
		weight1 = np.genfromtxt( "Weight_Save_MLP_weight1.csv", delimiter = ',' )
		weight2 = np.genfromtxt( "Weight_Save_MLP_weight2.csv", delimiter = ',' )
		print("Load weight ----- Done")
	else:
		for i in range(800):
			for j in range(756):
				weight1[ j, i ] = weight_init_variable( 'sigmoid' )
		for i in range(10):
			for j in range(800):
				weight2[ j, i ] = weight_init_variable( 'softmax' )
		print("weight init ----- Done")


''' extened function '''
def SAVE_WEIGHT_TO_CSV():
	np.savetxt( "Weight_Save_MLP_weight1.csv", weight1, delimiter = "," )
	np.savetxt( "Weight_Save_MLP_weight2.csv", weight2, delimiter = "," )
def PRINT_RATE( i, loop ):
	num = int( i / 30 )
	count = math.ceil( loop / num )
	count = count if count <= 31 else 31
	if loop % num == 0:
		print( "\033[8;3H{}".format( '#'*count ) )
	print( "\033[8;36H{}%".format( round( ( loop / i )*100 + 0.05, 2 ) ) )
def CHECK_ANS( loop ):
	if int( train_ans[ loop ][0] ) == np.argmax( output2 ):
		return True
	else:
		return False

''' Main '''
def Main( i ):
	global_initialization_variable()
	print("\033[7;2H{}".format('------------ Learning Rate ------------'))
	print("\033[8;2H{}".format('[                               ]'))
	
	for loop in range(i):
#		PRINT_RATE( i, loop )
		FEED_TRAIN_DATA( loop )
		TOTAL_FORWARD()
		TOTAL_REIVSE( loop )
		print( "\033[6;1H{}".format( HiddenError ) )
#		print( "\033[11;1H{}".format( output1 ) )
		if loop % 100 == 0:
			SAVE_WEIGHT_TO_CSV()
			print( train_ans[ loop ] )
			print( CHECK_ANS( loop ) )

#	SAVE_WEIGHT_TO_CSV()

if __name__ == "__main__":
	print( 'Train DataSet Sample -> ' + str( len( train_ans ) ) )
	Main( 2 )

