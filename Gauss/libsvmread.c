#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>

#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif 
#endif 
#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif

void exit_with_help()
{
	mexPrintf(
	"Usage: [label_vector, instance_matrix] = libsvmread('filename');\n"
	);
}

static void fake_answer(int nlhs, mxArray *plhs[])
{
	int i;
	for(i=0;i<nlhs;i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

static char *line;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename, int nlhs, mxArray *plhs[])
{
	int max_user_id=0, max_item_id=0;
	size_t l=0;
	FILE *fp = fopen(filename,"r");
	double *u_feats, *i_feats, *scores;
	mwIndex *u_ir, *u_jc, *i_ir, *i_jc;

	if(fp == NULL)
	{
		mexPrintf("can't open input file %s\n",filename);
		fake_answer(nlhs, plhs);
		return;
	}

	max_line_len = 1024;
	line = (char *) malloc(max_line_len*sizeof(char));

	while(readline(fp) != NULL)
	{
		char *u_id, *i_id, *score;
		int user_id, item_id; 

		u_id = strtok(line," \t");
		i_id = strtok(NULL," \t");
		score = strtok(NULL," \t");

		user_id = atoi(u_id);
		item_id = atoi(i_id);

		max_user_id = max(user_id, max_user_id);
		max_item_id = max(item_id, max_item_id);
		l++;
	}
	rewind(fp);

	// y
	plhs[0] = mxCreateDoubleMatrix(l, 1, mxREAL);
	// U
    plhs[1] = mxCreateSparse(max_user_id, l, l, mxREAL);
	// V
    plhs[2] = mxCreateSparse(max_item_id, l, l, mxREAL);

	//y pointer
	scores = mxGetPr(plhs[0]);
	//U pointer
	u_feats = mxGetPr(plhs[1]);
	u_ir = mxGetIr(plhs[1]);
	u_jc = mxGetJc(plhs[1]);
	//V pointer
	i_feats = mxGetPr(plhs[2]);
	i_ir = mxGetIr(plhs[2]);
	i_jc = mxGetJc(plhs[2]);

	int i = 0;
	while(readline(fp) != NULL)
	{
		char *u_id, *i_id, *score;
		int user_id, item_id;
		double d_score; 

		u_id = strtok(line," \t");
		i_id = strtok(NULL," \t");
		score = strtok(NULL," \t");

		user_id = atoi(u_id);
		item_id = atoi(i_id);
		d_score = atof(score);

		//Set y
		scores[i] = d_score;
		//Set U
		u_jc[i] = i;
		u_ir[i] = user_id-1;
		u_feats[i] = 1;
		//Set V
		i_jc[i] = i;
		i_ir[i] = item_id-1;
		i_feats[i] = 1;

		i++;
	}
	u_jc[i] = i;
	i_jc[i] = i;

	fclose(fp);
	free(line);

	for(i=1; i<3; i++)
	{
		mxArray *rhs[1], *lhs[1];
		rhs[0] = plhs[i];
		if(mexCallMATLAB(1, lhs, 1, rhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose problem\n");
			fake_answer(nlhs, plhs);
			return;
		}
		plhs[i] = lhs[0];
		mxDestroyArray(rhs[0]);
	}
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	char filename[256];

	if(nrhs != 1 || nlhs != 3)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);

	if(filename == NULL)
	{
		mexPrintf("Error: filename is NULL\n");
		return;
	}

	read_problem(filename, nlhs, plhs);

	return;
}

