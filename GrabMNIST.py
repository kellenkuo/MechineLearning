import csv

class data:
    def __init__( self ):
        self.count = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    def GrabTrain( self, number = 11 ):
        train_ans = []
        train_image = []
        with open( "mnist_train.csv", 'r' ) as train_file:
            sample = 892
            self.count = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

            for row in csv.reader( train_file ):
                if number == 11:
                    train_ans.append( row[:1][0] )
                    train_image.append( row[1:] )
                if number == 10:
                    if self.count[ int( row[:1][0] ) ] <= sample:
                        train_ans.append( row[:1][0] )
                        train_image.append( row[1:] )
                        self.count[ int( row[:1][0] ) ] += 1
        train_image, train_ans = self.conver_int( train_image, train_ans )
        return train_image, train_ans

    def GrabTest( self, number = 11 ):
        test_ans = []
        test_image = []
        with open( "mnist_test.csv", 'r' ) as test_file:
            sample = 892
            self.count = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

            for row in csv.reader( test_file ):
                if number == 11:
                    test_ans.append( row[:1][0] )
                    test_image.append( row[1:] )
                if number == 10:
                    if self.count[ int( row[:1][0] ) ] <= sample:
                        test_ans.append( row[:1][0] )
                        test_image.append( row[1:] )
                        self.count[ int( row[:1][0] ) ] += 1
        test_image, test_ans = self.conver_int( test_image, test_ans )
        return test_image, test_ans

    def conver_int( self, image, ans ):
        result_image = []
        result_ans = []
        for i in range( len( image ) ):
            tmp = []
            for j in range(784):
                tmp.append( int( image[i][j] ) )
            result_image.append( tmp )
        # change Ans to 0 and 1
        for i in range( len( ans ) ):
            tmp = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            tmp[ int( ans[i] ) ] = 1
            result_ans.append( tmp )
        return result_image, result_ans


    def recognize( self, row ):
        if str( row ) == '0':
            self.count[0] += 1
        elif str( row ) == '1':
            self.count[1] += 1
        elif str( row ) == '2':
            self.count[2] += 1
        elif str( row ) == '3':
            self.count[3] += 1
        elif str( row ) == '4':
            self.count[4] += 1
        elif str( row ) == '5':
            self.count[5] += 1
        elif str( row ) == '6':
            self.count[6] += 1
        elif str( row ) == '7':
            self.count[7] += 1
        elif str( row ) == '8':
            self.count[8] += 1
        elif str( row ) == '9':
            self.count[9] += 1
        else:
            pass

if __name__ == "__main__":
    call = data()
    test_image, test_ans = call.GrabTest(  )
    print( type( test_image[0][0] ) )

