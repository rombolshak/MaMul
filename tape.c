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

#define ROW 1
#define COL 2

double *row, *col, *res;
int procsNum, myRank;

void Init(double *A, double *B, int len) {
    int i, j;
    
    // pass columns and rows to others processors
    if (myRank == 0) {
	for (i = 1; i < procsNum; ++i) {
	    for (j = 0; j < len; ++j) {
		row[j] = A[j];
		col[j] = B[j * len + i];
	    }
	    MPI_Send(row, len, MPI_DOUBLE, i, ROW, MPI_COMM_WORLD);
	    MPI_Send(col, len, MPI_DOUBLE, i, COL, MPI_COMM_WORLD);
	}
    }
    if (myRank != 0) {
	MPI_Recv(row, len, MPI_DOUBLE, 0, ROW, MPI_COMM_WORLD, NULL);
	MPI_Recv(col, len, MPI_DOUBLE, 0, COL, MPI_COMM_WORLD, NULL);
    }
    
    // make the same for ourselves
    else
	for (i = 0; i < len; ++i) {
	    row[i] = A[i];
	    col[i] = B[i * len];
	}
}

void DoMult(int i, int len) {
    int j;    
    res[i] = 0;
    for (j = 0; j < len; ++j)
	res[i] += row[j] * col[j];
}

void Shift(int len) {
    MPI_Sendrecv_replace(col, len, MPI_DOUBLE, (myRank == 0) ? (len-1) : (myRank-1), COL, (myRank == len-1) ? (0) : (myRank + 1), COL, MPI_COMM_WORLD, NULL);
}

void TapeMult(double *A, double *B, double *C, int len) {
    int i;
    
    MPI_Comm_size(MPI_COMM_WORLD, &procsNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    
    if (procsNum != len) return; // number of processors must be the same the power of matrix is
    
    row = malloc(sizeof(double) * len);
    col = malloc(sizeof(double) * len);
    res = malloc(sizeof(double) * len);
    
    Init(A, B, len);
    
    for (i = 0; i < len; ++i) {
	DoMult(i, len);
	MPI_Barrier(MPI_COMM_WORLD);
	Shift(len);
	MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Gather(res, len, MPI_DOUBLE, C, len, MPI_DOUBLE, 0, MPI_COMM_WORLD); // getting result

    free(row);
    free(col);
    free(res);
    MPI_Barrier(MPI_COMM_WORLD);
}

