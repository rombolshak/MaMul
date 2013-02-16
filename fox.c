/*
 *  MaMul — matrix multiplication with MPI
 *  Copyright (C) 2013  Роман Большаков <rombolshak@russia.ru>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "fox.h"
#include "mpi.h"
#include "blockHelper.h"
#include <stdlib.h>

double *blockA_Const;

void FoxInit(double *A, double *B, double *C, int len) {
    blockA_Const = malloc(sizeof(double) * blockSize * blockSize);    
    BlocksInit(A, B, blockA_Const);
}

void DistributeA(int m) {
    int i, j, proc;
    
    for (i = 0; i < gridSize; ++i) {
	j = (i + m) % gridSize;
	proc = i * gridSize + j;
	if (myRank == proc) {
	    int k;
	    for (k = 0; k < gridSize; ++k) {
		if (k == j) continue;
		MPI_Send(blockA_Const, blockSize * blockSize, MPI_DOUBLE, i * gridSize + k, BLOCK_A, MPI_COMM_WORLD);
	    }
	    k = 0;
	    while (k < blockSize * blockSize) blockA[k] = blockA_Const[k++];
	}
	else if (myRow == i) {
	    MPI_Recv(blockA, blockSize * blockSize, MPI_DOUBLE, proc, BLOCK_A, MPI_COMM_WORLD, NULL);
	}
    }
}

void FoxMult(double *A, double *B, double *C, int len) {
    int m;

    if (!HelperInit(len)) return;
    FoxInit(A, B, C, len);
    
    for (m = 0; m < gridSize; ++m) {
	DistributeA(m);
	DoMult();
	ShiftB(1);
    }

    GetResultTo(C);    
    DisposeBlocks();
    free(blockA_Const);
}