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
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "tape.h"

int numProcs, myRank;

void genA(double *A, int len) {
    int i, j;
    for (i = 0; i < len; ++i)
	for (j = 0; j < len; ++j)
	    A[i * len + j] = (j % 2 == 0) ? -1 : 1;
}

void genB(double *B, int len) {
    int i, j;
    for (i = 0; i < len; ++i)
	for (j = 0; j < len; ++j)
	    B[i * len + j] = 1;
}

void genC(double *C, int len) {
    int i, j;
    for (i = 0; i < len; ++i)
	for (j = 0; j < len; ++j)
	    C[i * len + j] = 42;
}

int checkResult(double *C, int len) {
    int i, j;
    for (i = 0; i < len; ++i)
	for (j = 0; j < len; ++j)
	    if (((numProcs % 2 == 0) && (C[i * len + j] != 0)) ||
		((numProcs % 2 == 1) && (C[i * len + j] != -1)))
		return 0;
    return 1;
}

int main(int argc, char **argv) {
    double *A, *B, *C;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    A = malloc(sizeof(double) * numProcs * numProcs);
    B = malloc(sizeof(double) * numProcs * numProcs);
    C = malloc(sizeof(double) * numProcs * numProcs);
    
    if (myRank == 0) {
	genA(A, numProcs);
	genB(B, numProcs);}
	genC(C, numProcs);
    //}
    TapeMult(A, B, C, numProcs);
    printf("Check: %d\n", checkResult(C, numProcs));
    
    free(A);
    free(B);
    free(C);    
    return 0;
}
