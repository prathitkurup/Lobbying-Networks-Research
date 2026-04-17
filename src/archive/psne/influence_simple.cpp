//Compute equilibria of an influence game, without using divide and conquer

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

const int MAXNODES = 100;
const int optimist = 1;
char influentialNodesFileName[200];

int n;//# of nodes
const double fuzzy = 0;
int isEndPtNode[MAXNODES] = {0};
int endPtNodesList[MAXNODES] = {0};
int endPtCnt = 0;
int endPtAsg = 0;
char eqFileName[200]; //string is assigned in the main function 
char graphFileName[200]; //string is assigned in the main function using the command line argument

//////////////////////////////////////////////

const long INF = 2147483647;
long m; //# of arcs
double b[MAXNODES]; // threshold level
double graph[MAXNODES][MAXNODES];
int outDegree[MAXNODES], inDegree[MAXNODES];
int outList[MAXNODES][MAXNODES], inList[MAXNODES][MAXNODES];
FILE *fpout;
FILE *fpInfl;

long nLeavesVisited = 0;
long nNodesVisited = 0;

int getNextNode(int[], int[]);
int getLength(int[]);
int analysisMain();
void initStats();
void updateStats(int);
void printStats();
double NashProp();

double NashProp_SP();
long getN(int i, int j, int x_i, int x_j);
double getRho(int i, int j, int x_i, int x_j);

double cTime; //cputime

long min (long a, long b)
{
	return (a<b)?a:b;
}

double min (double a, double b)
{
	return (a<b)?a:b;
}

double max (double a, double b)
{
	return (a>b)?a:b;
}

int compareAscending(const void *a, const void *b)
{
   const double *da = (const double *) a;
   const double *db = (const double *) b;
 
   return (*da > *db) - (*da < *db);
}

int compareOutDegreeDescending(const void *a, const void *b)
{
   const double da = outDegree[*(const int*)a];
   const double db = outDegree[*(const int*)b];
 
   return -(da > db) + (da < db);
}


void getInput()
{
	//FILE *fpW = fopen("Senate_101_1.txt", "r+");
	//FILE *fpW = fopen("test_game.model", "r+");
	//FILE *fpW = fopen("viz_votes_congress110_session2_prep0neg_theta10_rho0.0027_methodSimulLR.txt", "r+");
	
	//FILE *fpW = fopen("viz_votes_congress110_session2_prep0neg_theta10_rho0.0038_methodSimulLR.txt", "r+");
	//FILE *fpW = fopen("Senate_108_1_0.01_W.txt", "r+");
   	//FILE *fpB = fopen("Senate_108_1_0.01_B.txt", "r+");

	//FILE *fpW = fopen("Senate_101_1_Adaboost_W.txt", "r+");
	//FILE *fpB = fopen("Senate_101_1_Adaboost_B.txt", "r+");

	//FILE *fpW = fopen("Random_Graph_Uniform_0.5.txt", "r+");
	
	FILE *fpW = fopen(graphFileName, "r");
	
	int i, j, temp;
	m = 0; //# of edges

	fscanf(fpW, "%d", &n);//first number: # of nodes
	/*if (n != temp)
	{
		printf("ERROR\n");
		getchar();
	}*/

	/*fscanf(fpW, "%d", &endPtCnt);//2nd number: # of endpt nodes -- indifferent
	
	//read the endpt nodes
	for (i = 0; i < endPtCnt; i++)
	{
		fscanf(fpW, "%d", &temp);
		isEndPtNode[temp] = 1;
		endPtNodesList[i] = temp;
	}*/

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (fscanf(fpW, "%lf", &graph[j][i]) != 1) //scan weight from j to i
			{
				printf("Invalid number of inputs...\n");
				exit(1);
			}
		}

		if (fscanf(fpW, "%lf", &b[i]) != 1) //scan threshold of i
		{
				printf("Invalid number of inputs...\n");
				exit(1);
		}
	}

	/*if (makeSymmetric == 1)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i != j && graph[j][i] != 0 && graph[i][j] == 0)
				{
					graph[i][j] = 0.000000000000000001;
				}
			}
		}
	}*/

	m = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			if (graph[j][i] != 0)
				m++;

		}
	}

	//printf("# edges = %ld\n", m);

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			outList[i][j] = -1;
			inList[i][j] = -1;
		}
	}

	int minInDeg = n, minOutDeg = n;
	int maxInDeg = -1, maxOutDeg = -1;
	double avgDeg = 0.0;

	for (i = 0; i < n ; i++)
	{
		outDegree[i] = 0;
		inDegree[i] = 0;
		//printf("From %d To: \t", i);
		for (j = 0; j < n; j++)
		{
			if (graph[i][j])
			{
				//printf("%d(%0.2lf)\t", j, graph[i][j]);
				outList[i][outDegree[i]] = j;
				outDegree[i]++;
				avgDeg++;
			}
		}
		//printf("Outdegree[%d] = %d\n", i, outDegree[i]);

		//printf("To %d: \t", i);
		for (j = 0; j < n; j++)
		{
			if (graph[j][i])
			{
				//printf("%d(%0.2lf)\t", j, graph[j][i]);
				inList[i][inDegree[i]] = j;
				inDegree[i]++;
			}
		}
		//printf("Indegree[%d] = %d\n\n", i, inDegree[i]);

		//if (outDegree[i] == 0)
		//	printf("%d\n", i);
		if (inDegree[i] < minInDeg)
			minInDeg = inDegree[i];
		if (outDegree[i] < minOutDeg)
			minOutDeg = outDegree[i];
		if (inDegree[i] > maxInDeg)
			maxInDeg = inDegree[i];
		if (outDegree[i] > maxOutDeg)
			maxOutDeg = outDegree[i];
	}
	
	if(n > 0)
		avgDeg /= (double)n;

	// printf("minInDeg = %d, minOutDeg = %d\nmaxInDeg = %d, maxOutDeg = %d\nAvgDeg = %0.4lf\n", 
	// 	minInDeg, minOutDeg, maxInDeg, maxOutDeg, avgDeg);

	fclose(fpW);
}

//data structure to measure the search space- should be 2^n at the end of backtracking
int bits[MAXNODES+1];
int highbit = -1; //highest bit on so far

/******************************************************************************************
Game:
f_i(x_{-i}) = \sum_{j \neq i} w_{ji} * x_j - b_i

if (f(x_{-i})) > 0 
	 x_i = 1;
else  (f(x_{-i})) < 0 
	x_i = -1;
else x_i \in {0,1}

*******************************************************************************************/

long equilibriumCount = 0;

//print a joint strategy to stdout
void print(int c[])
{
	int i;
	printf(":");
	for (i = 0; i < n; i++)
	{
		//if (c[i] == -INF)
		if (c[i] != 1 && c[i] != -1)
		{
			printf("#");
		}
		else 
			(c[i] == 1)?printf("Y"):printf("N");
	}
	printf("\n");
}

//print a joint strategy to the output file
void fprint(int c[])
{
	int i;
	//fprintf(fpout, "E ");
	for (i = 0; i < n; i++)
	{
		if (c[i] == -INF)
		{
			fprintf(fpout, "#\t");
			printf("ERROR IN JOINT ACTION (-INF)\n");
			exit(1);
		}
		else 
			(c[i] == 1)?fprintf(fpout, "1\t"):fprintf(fpout, "-1\t");
	}
	fprintf(fpout, "\n");
	fflush(fpout);
}

clock_t start = 0, end = 0;
double timeSpent = 0.0;

int testEquilibrium(int x[]) // x[] is a joint pure strategy
{
	int i, j;

	//check if x is an equilibrium
	for (i = 0; i < n; i++)
	{
		double sum = 0.0;
		for (j = 0; j < n; j++)
		{
			if (j == i)
				continue;
			sum += graph[j][i]*x[j];
		}

		//is x[i] a best response?
		if ((sum > b[i]+fuzzy*isEndPtNode[i] && x[i] == -1) || (sum < b[i]-fuzzy*isEndPtNode[i] && x[i] == 1)) //not an equilibrium
		{	
			//printf(": b[%d] = %lf, sum = %lf, x[%d] = %d\n", i, b[i], sum, i, x[i]);
			break;
		}
		//fprintf(fpout, ": b[%d] = %lf, sum = %lf, x[%d] = %d\n", i, b[i], sum, i, x[i]);
	}

	if (i == n) //yes, an equilibrium
	{
		if (start != 0)
		{
			//end = clock();
			//printf("\n\t\t|\n");
			//printf("CPU Time spent = %0.6lf\n", (double)(end - start)/(double)CLOCKS_PER_SEC);
			//printf("\t\t|\n");
		}
		//start = clock();
		equilibriumCount++;
		//printf("%ld.\t%ld\t%ld\n", equilibriumCount, nNodesVisited, nLeavesVisited);
		//print(x);
		fprint(x);
		/*if (equilibriumCount%100 == 0)
		{
			printf("%d\n", equilibriumCount);
			fflush(stdout);
		}*/
		return 1;
	}
	/*else
	{
		printf("No : ");
		print(x);
	}*/

	return 0;
}

//CAN BE MADE FASTER
int getLength(int c[])
{
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (c[i] == -INF)
			count++;
	}

	return n-count;
}


//select the next node using cut-set conditioning:
//i.e. select the one that will be most helpful in getting to a contradiction

//int nextNodeList[MAXNODES] = endPtNode;//{22, 25, 31, 34, 56, 64, 65, 78, 85, 86, 87, 93, 94, 95};
//int nNextNodes = endPtCnt;

int getNextNode(int c[], int returnVal[])
{
	int index_my = 0, i, j;

	//printf("-");

	/*if (getLength(c) == 0)
	{
		//select the first one -  the node with the max out-degree
		index = 0;

		for (i = 1; i < n; i++)
		{
			if (outDegree[i] > outDegree[index])
			{
				index = i;
			}
		}

		returnVal[0] = index;
		returnVal[1] = -1;
		return index;
	}*/
	if (getLength(c) < endPtCnt)
	{
		//select the first one -  the node with the max out-degree
		index_my = getLength(c);

		returnVal[0] = endPtNodesList[index_my];
		returnVal[1] = -1;
		return returnVal[0];
	}
	else
	{
		//select the next node according to largest influence on already selected nodes.
		double prevBestWt = -INF;
		double prevBestDiff = -INF;
		int prevBestNode = -1;
		int prevBestAction = -1;
		double sumInfluence[MAXNODES] = {0.0};
		int t;

		//calculate the current sum of influence at each selected node
		for (i = 0; i < n; i++)
		{
			if (c[i] == -INF)
				continue;
			sumInfluence[i] = 0.0;
			for (t = 0; t < inDegree[i]; t++)
			{
				j = inList[i][t];
				if (c[j] != -INF) //j already selected
				{
					sumInfluence[i] += graph[j][i] * c[j];
				}					
			}
		}

		for (i = 0; i < n; i++)
		{
			if (c[i] == -INF) //i not selected
			{
				continue;
			}

			//improve- iterate through i's inList only
			for (t = 0; t < inDegree[i]; t++)
			{
				j = inList[i][t];
				if (c[j] != -INF) //j already selected- ignore j
					continue;

				if (fabs(graph [j][i]) > prevBestWt)
				{
					prevBestWt = fabs(graph [j][i]);
					prevBestNode = j;
					if (optimist)
						prevBestAction = (graph[j][i] * b[i] > 0) ? (1) : -1;
					else
						prevBestAction = (graph[j][i] * b[i] < 0) ? (1) : -1;
					//prevBestAction = -1;
				}

				/*if ( fabs(graph[j][i]) - fabs(sumInfluence[i] - b[i]) > prevBestDiff) //we want the difference to be as large as possible
				{
					prevBestDiff = fabs(graph[j][i]) - fabs(sumInfluence[i] - b[i]);
					prevBestNode = j;
					prevBestAction = (graph[j][i] * (sumInfluence[i] - b[i]) > 0) ? (-1) : 1;
				}*/
			}
		}

		if (prevBestNode == -1) //disconnected component
		{
			for (i = 0; i < n; i++)
			{
				if (c[i] == -INF)
				{
					prevBestNode = i;
					prevBestAction = -1;
					break;
				}
			}
		}

		returnVal[0] = prevBestNode;
		returnVal[1] = prevBestAction;
		//printf("-%d. Next node = %d, action = %d\n", getLength(c), returnVal[0], returnVal[1]);
		return prevBestNode;
	}
}

//initialize backtracking
void preprocessBT(int c[])
{
	int i;
	for (i = 0; i < n; i++)
	{
		c[i] = -INF; //strategies are unassigned
		bits[i] = 0;
	}
	bits[i] = 0;

	nLeavesVisited = 0;
	nNodesVisited = 0;
}

//Include one more node into our consideration
int augment(int c[], int s[], int nodeNextActionPair[])
{
	int flag = 0;
	for (int i = 0; i < n; i++)
	{
		s[i] = c[i];
	}

	//printf("%d\n", getLength(c));
	if (getLength(c) == n)
	{
		flag = 0;
		return -1;
	}
	else
	{
		int nodeCurrentActionPair[2] = {0};
		int k;
		int len = getLength(c);
		
		k = getNextNode(c, nodeCurrentActionPair);
		
		c[k] = nodeCurrentActionPair[1];
		s[k] = nodeCurrentActionPair[1];
		nodeNextActionPair[0] = k;
		nodeNextActionPair[1] = (c[k] == -1)?1:(-1);
		return k;
	}
}

//Upadate search-space count: how much have we searched so far?
void updateCount(int c[])
{
	
	int p = 0, q = 0;

	for (p = 0; p < n; p++)
	{
		if (c[p] == -INF)
		{
			q++;
		}
	}

	for (p = q; ; p++)
	{
		if (bits[p] == 1)
		{
			bits[p] = 0;
		}
		else
		{
			bits[p] = 1;
			if (p > highbit)
			{
				highbit = p;
				//printf("-> %d", p); // only for 11111... patterns
			}
			break;
		}
	}
}

//int conflictCount = 0;

//Reject a partial set of strategies that can never lead to an equilibrium
//Use NashProp-like propagation for better results in rejection
int reject(int a[])
{
	int i, j;
	int c[MAXNODES];

	for (i = 0; i < n; i++)
	{
		c[i] = a[i];
	}

	//now find out which players' actions have become uniquely fixed (use NashProp)
	int changeMade = 1;
	int totalCh = 0;

	/*int conflictFlag = 0;
	if (getLength(c) > 0)
	{
		if (testConflict(c))//, cs.nodes[0], cs.actions[0]);
		{
			conflictFlag = 1;
			conflictCount++;
		}
	}

	if (conflictFlag)
		return 1;

	*/

	while(changeMade)
	{
		changeMade = 0;
		for (i = 0; i < n; i++)
		{
			if (c[i] == -INF)
			{
				//can i play 1?
				double sum = 0.0;
				for (j = 0; j < n; j++)
				{
					if (j == i)
						continue;
					if (c[j] == -INF)
					{
						//need to increase sum so that c[i] can play 1
						sum += fabs(graph[j][i]);
					}
					else 
					{
						sum += graph[j][i]*c[j];
					}
				}

				if (sum < b[i] + fuzzy*isEndPtNode[i])
				{
					c[i] = -1;
					changeMade = 1;
					totalCh++;
					break;
				}

				//can i play 0?
				sum = 0.0;
				for (j = 0; j < n; j++)
				{
					if (j == i)
						continue;
					if (c[j] == -INF)
					{
						//need to decrease sum so that c[i] can play -1
						sum -= fabs(graph[j][i]);
					}
					else 
					{
						sum += graph[j][i]*c[j];
					}
				}

				if (sum > b[i] - fuzzy*isEndPtNode[i])
				{
					c[i] = 1;
					changeMade = 1;
					totalCh++;
					break;
				}				
			}
		}
	}

	//printf("TotalCh = %d\n", totalCh);

	for (i = 0; i < n; i++)
	{
		if (c[i] == -INF)
			continue;

		//int cArray[MAXNODES]; //neighborhood array for conflict set

		if (c[i] == 1)
		{
			double sum = 0.0;

			//cArray[i] = 1;
			
			for (j = 0; j < n; j++)
			{
				if (j == i)
					continue;
				if (c[j] == -INF)
				{
					//need to increase sum so that c[i] can play 1
					sum += fabs(graph[j][i]);
				}
				else 
				{
					sum += graph[j][i]*c[j];
				}
				
				//if (graph[j][i])
				//{
				//	cArray[j] = c[j];
				//}
				//else cArray[j] = -INF;
			}

			if (sum < b[i] - fuzzy*isEndPtNode[i])
			{
				//fprintf(fpout, "REJECT: sum = %0.4lf, b[%d] = %0.4lf, x[%d] = %d\n", sum, i, b[i], i, c[i]);		
				//fprintf(fpout, "REJECT: ");
				//fprint(c);

				/*ConflictSet cs = makeConflictSet(cArray);
				if (cs.length > 0)
					updateConflictSets(cs);
				updateCount(c);*/
				//printf("(%d)", getLength(c));
				return 1;
			}
		}
		else if (c[i] == -1)
		{
			double sum = 0.0;

			//cArray[i] = -1;

			for (j = 0; j < n; j++)
			{
				if (j == i)
					continue;
				if (c[j] == -INF)
				{
					//need to decrease sum so that c[i] can play -1
					sum -= fabs(graph[j][i]);
				}
				else 
				{
					sum += graph[j][i]*c[j];
				}

				//if (graph[j][i])
				//{
				//	cArray[j] = c[j];
				//}
				//else cArray[j] = -INF;
			}

			if (sum > b[i] + fuzzy*isEndPtNode[i])
			{
				//fprintf(fpout, "REJECT: sum = %0.4lf, b[%d] = %0.4lf, x[%d] = %d\n", sum, i, b[i], i, c[i]);		
				//fprintf(fpout, "REJECT: ");
				//fprint(c);
				/*ConflictSet cs = makeConflictSet(cArray);
				if (cs.length > 0)
					updateConflictSets(cs);
				updateCount(c);*/
				//printf("(%d)", getLength(c));
				return 1;
			}
		}
	}

	return 0;
}

void backTrack(int d[])
{
	int c[MAXNODES] = {0};
	for (int i = 0; i < n; i++)
	{
		c[i] = d[i];
	}

	nNodesVisited++;
	if (getLength(c) == n) //We have obtained a complete joint strategy
	{
		nLeavesVisited++;
		updateCount(c);
		testEquilibrium(c);
	}
	else if (reject(c))
	{
		return;
	}

	int s[MAXNODES] = {0};
	int nodeNextActionPair[2] = {0};
	int augIndex = augment(c, s, nodeNextActionPair);
	if (augIndex >= 0) 
	{
		if (isEndPtNode[augIndex])
		{
			//fprintf(fpout, "NNNNNN\n");
		}
		if (getLength(c) == endPtCnt)
		{
			fprintf(fpout, "N\n");
		}

		backTrack(s);
		
		s[augIndex] = nodeNextActionPair[1];
		//printf("%d's next action = %d\n", augIndex, nodeNextActionPair[1]);
		if (isEndPtNode[augIndex])
		{
			//fprintf(fpout, "NNNNNN\n");
		}
		if (getLength(c) == endPtCnt)
		{
			fprintf(fpout, "N\n");
		}
		backTrack(s);
	}
}


/////////////////////////// BRUTE-FORCE EQ COMPUTATION ///////////////
int nextX(int x[])
{
	int i;
	int sumOne = 0;

	for (i = 0; i < n; i++)
	{
		if (x[i] == 1)
			sumOne++;
	}

	if (sumOne == n)
		return 0;

	for (i = 0; i < n; i++)
	{
		if (x[i] == 1)
		{
			x[i] = -1;
		}
		else
		{
			x[i] = 1;
			break;
		}
	}
	return 1;
}


void bruteForce()
{
	int x[MAXNODES] = {0};
	int i;

	equilibriumCount = 0;

	for (i = 0; i < n; i++)
	{
		x[i] = -1;
	}

	do
	{
		testEquilibrium(x);

	} while (nextX(x));
}

long bruteForceFixedAction(int player, int actionOfPlayer)
{
	int x[MAXNODES] = {0};
	int i;

	equilibriumCount = 0;

	for (i = 0; i < n; i++)
	{
		x[i] = -1;
	}

	long count = 0;
	do
	{
		if (x[player] == actionOfPlayer)
			count += testEquilibrium(x);

	} while (nextX(x));

	return count;
}

int main( int argc, char *argv[] )  
{
	if( argc == 2 )
	{
		char fArg[200];

		strcpy(fArg, argv[1]);
		strcpy(graphFileName, fArg);

		//get rid of ".txt", because it will be added after "_equilibria"
		fArg[strlen(fArg)-4] = 0;
		strcpy(eqFileName, strcat(fArg, "_equilibria.txt")); //output file
	}

	clock_t mainStart = 0, mainEnd = 0;
	
	mainStart = clock();

	fpout = fopen(eqFileName, "w");
	getInput();
	
	if (n == 0)
		return 0;
	
	/*if (endPtCnt == 0)
	{
		fprintf(fpout, "N\n");
	}*/

	int c[MAXNODES];
	preprocessBT(c);
	backTrack(c);

	//bruteForce(); //instead of the above two lines, you can also do brute force computation to check the correctness. Don't use both bruteforce and backtracking search.
	
	mainEnd = clock();
	double t = (double)(mainEnd - mainStart)/(double)CLOCKS_PER_SEC;	

	printf("Time = %lf sec\n", t);

	//printf("Time = %lf sec\n#nodes visited = %ld\n", t, nNodesVisited);

	//fprintf(fpout, "END\n");
	
	//printf("Above: N denotes action -1 and Y represents action 1\n");
	// printf("%s: #EQUILIBRIA = %ld\n", graphFileName, equilibriumCount);

	printf("# Equilibria = %ld\n", equilibriumCount);

	return 0;
}
