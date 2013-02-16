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
#include "fox.h"
#include "cannon.h"

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
	for (j = 0; j < len; ++j) {
	    //printf("C[%d, %d] = %f\n", i, j, C[i * len + j]);
	    if (((numProcs % 2 == 0) && (C[i * len + j] != 0)) ||
		((numProcs % 2 == 1) && (C[i * len + j] != -1)))
		return 0;
	}
    return 1;
}

int main(int argc, char **argv) {
    double *A, *B, *C;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    A = malloc(sizeof(double) * numProcs * numProcs * 16);
    B = malloc(sizeof(double) * numProcs * numProcs * 16);
    C = malloc(sizeof(double) * numProcs * numProcs * 16);
    
    if (myRank == 0) {
	genA(A, numProcs * 4);
	genB(B, numProcs * 4);}
	genC(C, numProcs * 4);
    //}
    //TapeMult(A, B, C, numProcs);
    //FoxMult(A, B, C, numProcs * 4);
    CannonMult(A, B, C, numProcs * 4);
    printf("Check: %d\n", checkResult(C, numProcs * 4));
    
    free(A);
    free(B);
    free(C);    
    return 0;
}
