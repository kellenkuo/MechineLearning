from PIL import Image
import numpy as np
import csv
import random
import method

# int16 = short
# int32 = int
# int64 = long

# iMage range 0 - 255
# csv file = Ans + color * 784

precision = 6
judgment = 50
Learning_Rate = 0.019
Load_Weight = False
def weight_init_variable():
	return round( random.uniform( 0.1, 1 ), precision )

train_image = []
train_ans = []

test_image = []
test_ans = []

with open( "mnist_train.csv" ) as train_file:
	count = 0
	for row in csv.reader( train_file ):
		train_ans.append( row[:1] )
		train_image.append( row[1:] )
		count += 1
		print( "\033[2;1HRead train CSV file ----- {0}".format( count ) )
	print( "\033[2;1HRead train CSV file ----- {0}".format( 'All Done' ) )

with open( "mnist_test.csv" ) as test_file:
	count = 0
	for row in csv.reader( test_file ):
		test_ans.append( row[:1] )
		test_image.append( row[1:] )
		count += 1
		print( "\033[3;1HRead test CSV file ----- {0}]".format( count ) )
	print( "\033[3;1HRead test CSV file ----- {0}]".format( 'All Done' ) )

# predict
# create Target
target = np.zeros( (10,1), dtype = np.int16 )
target_test = 0

# create input
# do reshape
Input = np.zeros( (1,784), dtype = np.int16 )
print("Create Input ----- Done")
# create weight
weight = np.zeros( (10,784), dtype = np.float64 )
print("Create weight ----- Done")
# create output
output = np.zeros( (10,1), dtype = np.int16 )
print("Create output ----- Done")

# initialization_variable
def global_initialization_variable():
	global weight
	# weight init to 0.001
	if Load_Weight:
		weight = np.genfromtxt( "Weight_Save_2.csv", delimiter = ',' )
		print("Load weight ----- Done")
	else:
		for i in range(10):
			for j in range(784):
				weight[ i, j ] = weight_init_variable()
		print("weight init ----- Done")

# Load DATA To Input
def LOAD_DATA_SET( i ):
	global Input
	tmp = []
	for load in train_image[i]:
		tmp.append( load )
	for i in range(784):
		Input[ 0, i ] = tmp[i]
#	print( Input )

# Load sample target
def LOAD_TARGET(i):
	global target
	tmp = []
	for load in train_ans[i]:
		ans = int( load )
#	print( ans )
	for i in range(10):
		if i == ans:
			target[ i, 0 ] = 1
		else:
			target[ i, 0 ] = 0
#	print( target )

''' tanh Method '''
# tanh_derivative limit = 0.4199743
# NEED TO DO DATA VANISH
def WEIGHT_CALCULATE_tanh( loop ):
	global output, weight
	for ans in train_ans[loop]:
		i = int( ans )
	result = 0
	for j in range(784):
		result += Input[ 0, j ] * 0.01 * weight[ i, j ]
	result = result / 784.0
	correct = method.tanh( result )
	revise = method.tanh_derivative( correct )
	for j in range(784):
		weight[ i, j ] += Input[ 0, j ] * 0.01 * revise * Learning_Rate


''' TURE OR FALSE Method '''
def WEIGHT_CALCULATE_BOOL( loop ):
	global output, weight
	for ans in train_ans[loop]:
		i = int( ans )
	for j in range(784):
		if Input[ 0, j ] > judgment:
			weight[ i, j ] += Input[ 0, j ] * 0.004 * Learning_Rate
		else:
			weight[ i, j ] -= Input[ 0, j ] * 0.004 * Learning_Rate


''' test data correct rate '''
# Load test_DATA To Input
def LOAD_TEST_DATA_SET( i ):
	global Input
	tmp = []
	for load in test_image[i]:
		tmp.append( load )
	for i in range(784):
		Input[ 0, i ] = tmp[i]

# Load test target
def LOAD_TEST_TARGET(i):
	global target_test
	tmp = []
	for load in test_ans[i]:
		ans = int( load )
	target_test = ans

# nutral network calc
def NETWORK_CALCULATE():
	result = []
	for i in range(10):
		total = 0
		for j in range(784):
			total = total + Input[ 0, j ] * 0.01 * weight[ i, j ]
		result.append( total )
# find max value in result
	Max_index = result.index( max( result ) )
	if Max_index == target_test:
		return True
	else:
		return False
''' do testing data '''
def DO_TESTING_DATA( loop ):
	LOAD_TEST_DATA_SET( loop )
	LOAD_TEST_TARGET( loop )
	check = NETWORK_CALCULATE()
	return check

''' SAVE WEIGHT '''
def SAVE_WEIGHT_TO_CSV():
	np.savetxt( "Weight_Save_2.csv", weight, delimiter = "," )

''' Correct Rate '''
def RATE( result ):
	tmp_all = 0
	tmp_correct = 0
	for i in result:
		tmp_all = tmp_all + 1
		if i == True:
			tmp_correct = tmp_correct + 1
	calc = round( tmp_correct / tmp_all, 2 )
	return calc

''' Let DATA VANISH '''
def DO_WEIGHT_VANISH():
	global weight
	value = 0
	for i in range(10):
		for j in range(784):
			if value < weight[ i, j ]:
				value = weight[ i, j ]
	for i in range(10):
		for j in range(784):
			weight[ i, j ] = round( weight[ i, j ] / value, precision )

def Main():
	global_initialization_variable()

	for loop in range(10000):
		LOAD_DATA_SET( loop )
		LOAD_TARGET( loop )
		WEIGHT_CALCULATE_tanh( loop )

		if loop % 1000 == 999:
			DO_WEIGHT_VANISH()
			tmp = []
			for test in range(100):
				result = DO_TESTING_DATA( test )
				tmp.append( result )
			print( RATE( tmp ) , end = '' )
			print( "%" )

		if loop == 5000:
			Learning_Rate = 0.15
	SAVE_WEIGHT_TO_CSV()

if __name__ == "__main__":
	try:
		Main()
	except KeyboardInterrupt():
		exit()
