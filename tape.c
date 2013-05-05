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
#include "tape.h"
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define ROW 1
#define COL 2

double *row, *col, *res;
int procsNum, myRank, k;

void TapeInit(double *A, double *B, int len) {
    int i, j;
    //printf("Enter TapeInit\n");
    // pass columns and rows to others processors
    if (myRank == 0) {
	for (i = procsNum - 1; i >= 0; --i) {
	    for (j = 0; j < k * len; ++j) {
		int rowInB = (int)floor(1.0 * j / k);
		row[j] = A[i * k * len + j];
		col[j] = B[rowInB * len + i * k + (j % k)];
	    }
	    if (i != 0) {
		MPI_Send(row, k * len, MPI_DOUBLE, i, ROW, MPI_COMM_WORLD);
		MPI_Send(col, len * k, MPI_DOUBLE, i, COL, MPI_COMM_WORLD);
	    }
	}
    }    
    else {
	MPI_Recv(row, k * len, MPI_DOUBLE, 0, ROW, MPI_COMM_WORLD, NULL);
	MPI_Recv(col, len * k, MPI_DOUBLE, 0, COL, MPI_COMM_WORLD, NULL);
    }
    //printf("Exit TapeInit\n");
}

void TapeDoMult(int i, int len) {
    int j, l, t, cell;
    
    cell = (procsNum + myRank - i) % procsNum;
    for (j = 0; j < k; ++j) // строка в полосе А
	for (l = 0; l < k; ++l) // столбец в полосе В
	    for (t = 0; t < len; ++t)
		res[j * len + cell * k + l] += row[j * len + t] * col[t * k + l];
}

void Shift(int len) {
    MPI_Sendrecv_replace(col, len * k, MPI_DOUBLE, (myRank == 0) ? (procsNum-1) : (myRank-1), COL, (myRank == procsNum-1) ? (0) : (myRank + 1), COL, MPI_COMM_WORLD, NULL);
}

void TapeMult(double *A, double *B, double *C, int len) {
    int i;
    
    MPI_Comm_size(MPI_COMM_WORLD, &procsNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    k = len / procsNum; // строк в одной полосе
        
    row = malloc(sizeof(double) * k * len);
    col = malloc(sizeof(double) * len * k);
    res = malloc(sizeof(double) * k * len);
    bzero(res, sizeof(double) * k * len);
    //printf("Malloc success\n");
    
    TapeInit(A, B, len);
    
    //printf("Begin main loop\n");
    for (i = 0; i < procsNum; ++i) {
	TapeDoMult(i, len);
	Shift(len);
    }
    //printf("End main loop\n");
    MPI_Gather(res, k * len, MPI_DOUBLE, C, k * len, MPI_DOUBLE, 0, MPI_COMM_WORLD); // getting result

    free(row);
    free(col);
    free(res);
    MPI_Barrier(MPI_COMM_WORLD);
}

