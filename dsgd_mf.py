import sys
import scipy.sparse as sps
import numpy as np
import random
import math
import csv
from pyspark import SparkContext, SparkConf
'''
    Loads the data into a sparse matrix and returns it
'''
def LoadSparseMatrix(ratingsFile):
	tArray = np.loadtxt(ratingsFile, delimiter=",", usecols=(0, 1, 2))	# Consider only first three columns
	(row, col, val) = (tArray[:, 0] - 1, tArray[:, 1] - 1, tArray[:, 2])
	rsize = np.max(row) + 1	# Row size
	csize = np.max(col) + 1	# Column size
	return sps.csr_matrix((val, (row, col)), shape=(rsize, csize), dtype=float)

'''
    Computes the dot product of two vectors or matrices
'''
def dotProduct(M):
    return np.sum(np.multiply(M, M))

'''
    SGD operations happen in this function
'''
def computeGradient(block):
    Vmini, Wmini, Hmini = block[0], block[1], block[2]
    blockEntryRow, blockExitRow, blockEntryCol, blockExitCol = block[3], block[4], block[5], block[6]
    globalIter = block[7]
    rowZ, colZ = Vmini.nonzero()                        # Getting the non-zero values

    total = rowZ.size 
    if total == 0:
    	return (Wmini, Hmini, blockEntryRow, blockExitRow, blockEntryCol, blockExitCol, globalIter + iteration)
	
    iteration = 0

    while iteration < total:
        #randomIndex = random.randint(0, total-1)        # Get a random rating
        r = rowZ[iteration]
        c = colZ[iteration]
        W = Wmini[r, :]
        H = Hmini[:, c]
        learningRate = math.pow((100 + globalIter +  iteration), -betaValue)
    
        temp = 2.0 * (Vmini[r, c] - W.dot(H))
        
        vrnonzeros = 1.0 * Vmini[r, :].nonzero()[0].size    # Getting N_i*
        vcnonzeros = 1.0 * Vmini[:, c].nonzero()[0].size    # Getting N_*j

        W += (learningRate * temp * H.T) - (2.0 * learningRate * (lambdaValue/vrnonzeros) * W)
        H += (learningRate * temp * Wmini[r, :].T) - (2.0 * learningRate * (lambdaValue/vcnonzeros) * H)
        Wmini[r, :] = W                                     # Updating W[i, :]
        Hmini[:, c] = H                                     # Updating H[:, j]
        iteration += 1
    
    # Return the updated metadata
    return (Wmini, Hmini, blockEntryRow, blockExitRow, blockEntryCol, blockExitCol, globalIter + iteration)
    
betaValue = 0.0
lambdaValue = 0.0
#mseOut = open("mse_out.txt", "w")

'''
    Function for writing both W and H matrices
'''
def writeOutput(npArr, outputFile):
    np.savetxt(outputFile, npArr, delimiter=",")
'''
    Calculates the normalized mean squared error at each iteration
'''
def calculateMSE(iteration, pred, V, select):
    global mseOut
    diff = V - pred
    mse = 0.0
    for row, col in select:
        mse += diff[row, col] * diff[row, col]
    mseOut.write("Mean Squared Error: Iteration {0} {1}\n".format(iteration+1, mse/len(select)))
'''
   Returns the loss computed after each iteration
'''
def calculateLoss(pred, V, select, W, H):
    diff = V - pred
    loss = 0.0
    for row, col in select:
        loss += diff[row, col] * diff[row, col]
    loss = loss / len(select)
    loss += dotProduct(W) + dotProduct(H)
    return loss
    
def main():
    global betaValue, lambdaValue
    noOfFactors = int(sys.argv[1])          # Number of factors
    noOfWorkers = int(sys.argv[2])          # Number of workers
    noOfIterations = int(sys.argv[3])       # Number of iterations
    betaValue = float(sys.argv[4])          # Beta value (decay parameter)
    lambdaValue = float(sys.argv[5])        # Regularization parameter
    inputPath = sys.argv[6]                 # Path to input (it can be a directory or a file)
    outputWPath = sys.argv[7]               # Output file for writing W factored matrix in csv format
    outputHPath = sys.argv[8]               # Output file for writing H factored matrix in csv format
    
    conf = SparkConf().setAppName("DSGD")#.setMaster("local[{0}]".format(noOfWorkers))
    sc = SparkContext(conf=conf)
    data = LoadSparseMatrix(inputPath)  # Loads the sparse matrix into the memory
    noOfUsers = data.shape[0]
    noOfMovies = data.shape[1]
    remainRow, remainCol = noOfUsers%noOfWorkers, noOfMovies%noOfWorkers
    if remainRow > 0:
        remainRow = 1
    if remainCol > 0:
        remainCol = 1
        
    blockSize = ((noOfUsers/noOfWorkers) + remainRow, (noOfMovies/noOfWorkers) + remainCol) # Size of each block in a stratum
    W = np.random.random_sample((noOfUsers, noOfFactors))   # Initialize W
    H = np.random.random_sample((noOfFactors, noOfMovies))  # Initialize H
   
    stratumIndices = {}
    for stratum in xrange(noOfWorkers):                     # Creating stratum
        blocks = [] 
        for worker in xrange(noOfWorkers):                  # Creating blocks in a stratum
            blockEntryRow, blockExitRow = worker * blockSize[0], (worker + 1) * blockSize[0]
            blockEntryCol, blockExitCol = (worker + stratum) * blockSize[1], (stratum + worker + 1) * blockSize[1]
            if blockEntryCol > noOfMovies:
                blockEntryCol = (blockEntryCol % noOfMovies) - 1
                blockExitCol = blockEntryCol + blockSize[1]
            if blockExitRow > noOfUsers:
                blockExitRow = noOfUsers
            if blockExitCol > noOfMovies:
                blockExitCol = noOfMovies
            blocks.append((blockEntryRow, blockExitRow, blockEntryCol, blockExitCol))
        stratumIndices[stratum] = blocks
    prevLoss = -1.0
    for iteration in xrange(noOfIterations):                # Number of times it goes through the training data
        globalIter = 0

        for stratum in xrange(noOfWorkers):                     # Creating stratum
            blocks = [] 
            for worker in xrange(noOfWorkers):                  # Creating blocks in a stratum
                block = stratumIndices[stratum][worker]
                blockEntryRow, blockExitRow, blockEntryCol, blockExitCol = block[0], block[1], block[2], block[3]
                Vmini = data[blockEntryRow:blockExitRow, blockEntryCol:blockExitCol]
                Wmini = W[blockEntryRow:blockExitRow, :]
                Hmini = H[:, blockEntryCol:blockExitCol]
                # Preparing meta data for blocks 
                blocks.append((Vmini, Wmini, Hmini, blockEntryRow, blockExitRow, blockEntryCol, blockExitCol, globalIter)) 
            result = sc.parallelize(blocks, noOfWorkers).map(lambda x: computeGradient(x)).collect()    # Run SGD in parallel
            for r in result:            # Updating W, H from the partial results of each worker
                blockEntryRow, blockExitRow, blockEntryCol, blockExitCol = r[2], r[3], r[4], r[5]
                globalIter += r[6]      # Updating global iterations from each SGD
                W[blockEntryRow:blockExitRow, :] = r[0] # Updating W
                H[:, blockEntryCol:blockExitCol] = r[1] # Updating H
	loss = calculateLoss(W.dot(H), data, select, W, H)
    	if np.fabs(loss - prevLoss) < 0.00001:
    		break
    	else:
    		prevLoss = loss
    #calculateMSE(iteration, W.dot(H), data, select)

    writeOutput(W, outputWPath)
    writeOutput(H, outputHPath)
    #global mseOut
    #mseOut.close()
    sc.stop()           # Shutting down spark context
    
if __name__ == "__main__":
    main()
