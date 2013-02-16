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
#include "cannon.h"
#include "blockHelper.h"


void CannonInit(double *A, double *B, int len) {
    int i;
    
    BlocksInitDefault(A, B);

    for (i = 1; i < gridSize; ++i) {
	if (myRow == i) ShiftA(i);
	if (myCol == i) ShiftB(i);
    }
}

void CannonMult (double *A, double *B, double *C, int len) {
    int i;
    
    if (!HelperInit(len)) return;
    CannonInit(A, B, len);

    for (i = 0; i < gridSize; ++i) {
	DoMult();
	ShiftA(1);
	ShiftB(1);
    }

    GetResultTo(C);
    DisposeBlocks();
}