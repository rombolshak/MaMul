#ifndef _BLOCK_HELPER_H
#define _BLOCK_HELPER_H

#define BLOCK_A 1
#define BLOCK_B 2

int myRow, myCol, myRank, procsnum, gridSize, blockSize;
double *blockA, *blockB, *blockC;

int HelperInit(int len);
void BlocksInit(double *A, double *B, double *customBlockA);
void BlocksInitDefault(double *A, double *B);
void ShiftA(int i);
void ShiftB(int i);
void DoMult();
void GetResultTo(double *C);
void DisposeBlocks();

#endif