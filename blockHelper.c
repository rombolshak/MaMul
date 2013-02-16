#include "blockHelper.h"
#include "mpi.h"
#include "math.h"
#include <stdlib.h>

int HelperInit (int len) {
    MPI_Comm_size(MPI_COMM_WORLD, &procsnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    gridSize = (int)(sqrt(procsnum));
    if (procsnum != gridSize * gridSize) return 0;
    
    myRow = (int)floor(myRank * 1.0 / gridSize);
    myCol = myRank % gridSize;
    blockSize = len / gridSize; // размер одного блока
    if (blockSize * gridSize != len) return 0;
    return 1;
}

void getBlocks(double *A, double *B, int row, int col) {
    int i, j;
    int len = blockSize * gridSize;
    
    for (i = 0; i < blockSize; ++i)
	for (j = 0; j < blockSize; ++j) {
	    blockA[i * blockSize + j] = A[(row * gridSize + i) * len + (col * gridSize + j)];
	    blockB[i * blockSize + j] = B[(row * gridSize + i) * len + (col * gridSize + j)];
	}
}

void BlocksInit(double* A, double* B, double *customBlockA)
{
    int row, col;
    
    blockA = malloc(sizeof(double) * blockSize * blockSize);
    blockB = malloc(sizeof(double) * blockSize * blockSize);
    blockC = malloc(sizeof(double) * blockSize * blockSize);
    if (customBlockA == NULL) customBlockA = blockA;
    
    if (myRank == 0) {
	for (row = 0; row < gridSize; ++row)
	    for (col = 0; col < gridSize; ++col) {
		if ((row == 0) && (col == 0)) continue;
		getBlocks(A, B, row, col);
		MPI_Send(customBlockA, blockSize * blockSize, MPI_DOUBLE, row * gridSize + col, BLOCK_A, MPI_COMM_WORLD);
		MPI_Send(blockB, blockSize * blockSize, MPI_DOUBLE, row * gridSize + col, BLOCK_B, MPI_COMM_WORLD);
	    }
    }
    
    if (myRank != 0) {
	MPI_Recv(customBlockA, blockSize * blockSize, MPI_DOUBLE, 0, BLOCK_A, MPI_COMM_WORLD, NULL);
	MPI_Recv(blockB, blockSize * blockSize, MPI_DOUBLE, 0, BLOCK_B, MPI_COMM_WORLD, NULL);
    }
    else {
	getBlocks(A, B, 0, 0);
    }
    
    for (row = 0; row < blockSize; ++row)
	for (col = 0; col < blockSize; ++col)
	    blockC[row * blockSize + col] = 0;
}

void BlocksInitDefault(double* A, double* B)
{
    BlocksInit(A, B, NULL);
}

void ShiftA(int i) {
    MPI_Sendrecv_replace(blockA, blockSize * blockSize, MPI_DOUBLE,
			 (myCol - i) < 0 ? 		(myRow * gridSize + (myCol + gridSize - i)) : (myRow * gridSize + (myCol - i)), BLOCK_A,
			 (myCol + i) >= gridSize ? 	(myRow * gridSize + (myCol + i - gridSize)) : (myRow * gridSize + (myCol + i)), BLOCK_A, MPI_COMM_WORLD, NULL);
}

void ShiftB(int i) {
    MPI_Sendrecv_replace(blockB, blockSize * blockSize, MPI_DOUBLE,
			 (myRow - i) < 0 ? 		((myRow + gridSize - i) * gridSize + myCol) : ((myRow - i) * gridSize + myCol), BLOCK_B,
			 (myRow + i) >= gridSize ? 	((myRow + i - gridSize) * gridSize + myCol) : ((myRow + i) * gridSize + myCol), BLOCK_B, MPI_COMM_WORLD, NULL);
}

void DoMult() {
    int i, j, k;
    
    for (i = 0; i < blockSize; ++i)
	for (j = 0; j < blockSize; ++j)
	    for (k = 0; k < blockSize; ++k)
		blockC[i * blockSize + j] += blockA[i * blockSize + k] * blockB[k * blockSize + j];
}

void InsertC(double *C, int row, int col) {
    int i, j, len = blockSize * gridSize;
    
    for (i = 0; i < blockSize; ++i)
	for (j = 0; j < blockSize; ++j) {
	    //printf("C[%d, %d] = %f\n", (row * blockSize + i),(col * blockSize + j),blockC[i * blockSize + j]);
	    C[(row * blockSize + i) * len + (col * blockSize + j)] = blockC[i * blockSize + j];
	}
}

void GetResultTo(double *C) {
    if (myRank != 0)
	MPI_Send(blockC, blockSize * blockSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else {
	int row, col;
	for (row = 0; row < gridSize; ++row)
	    for (col = 0; col < gridSize; ++col) {
		if ((row != 0) || (col != 0))
		    MPI_Recv(blockC, blockSize * blockSize, MPI_DOUBLE, row * gridSize + col, 0, MPI_COMM_WORLD, NULL);
		InsertC(C, row, col);
	    }
    }
}

void DisposeBlocks()
{    
    free(blockA);
    free(blockB);
    free(blockC);
}
