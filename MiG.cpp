#include <math.h>
#include "mex.h"
#include <string.h>

/// Compute the function value of average regularized logistic loss
/// *w - tesp point
/// *Xt - data matrix
/// *y - set of labels
/// lambda - regularization parameter
/// n - number of training examples
/// d - dimension of the problem
double compute_function_value(double* w, double *Xt, double *y, double lambda,
	long n, long d)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = 0; j < d; j++) {
			tmp += Xt[i*d + j] * w[j];
		}
		value += log(1 + exp(y[i] * tmp));
	}
	value = value / n;
	for (long j = 0; j < d; j++) {
		value += lambda* w[j] * w[j] / 2;
	}
	return value;
}

/// compute_sigmoid computes the derivative of logistic loss,
/// i.e. ~ exp(x) / (1 + exp(x))
/// *x - pointer to the first element of the training example
///		 e.g. Xt + d*i for i-th example
/// *w - test point
/// y - label of the training example
/// d - dimension of the problem
double compute_sigmoid(double *x, double *w, double y, long d)
{
	double tmp = 0;
	// Inner product
	for (long j = 0; j < d; j++) {
		tmp += w[j] * x[j];
	}
	tmp = exp(y * tmp);
	tmp = y * tmp / (1 + tmp);
	return tmp;
}

/// compute_full_gradient computes the gradient 
/// of the entire function. Gradient is changed in place in g
/// *Xt - data matrix; examples in columns!
/// *w - test point
/// *y - set of labels
/// *g - gradient; updated in place; input value irrelevant
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
void compute_full_gradient(double *Xt, double *w, double *y,
	double *g, long n, long d, double lambda)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_sigmoid(Xt + d*i, w, y[i], d);
		for (long j = 0; j < d; j++) {
			g[j] += Xt[d*i + j] * sigmoid;
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
		g[i] += lambda * w[i];
	}
}

void compute_momentum(double *tmp, double *w, double *wold,
	double theta, long d) {
	for (long j = 0; j < d; j++)
	{
		tmp[j] = theta * w[j] + (1 - theta) * wold[j];
	}
}

/// Update the test point *w in place once you have everything prepared
/// *x - training example
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *gold - full gradient computed at point *wold
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// d - dimension of the problem
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
void update_test_point_dense(double *x, double *w, double *wold, double *temp,
	double *gold, double sigmoid, double sigmoidold,
	long d, double stepSize, double lambda)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (gold[j] + x[j] *
			(sigmoid - sigmoidold) + lambda * (temp[j] - wold[j]));
	}
}

/// Compute the function value of average regularized logistic loss
/// *w - tesp point
/// *Xt - data matrix
/// *y - set of labels
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
double compute_function_value_sparse(double* w, double *Xt, double *y, double lambda,
	long n, long d, mwIndex *ir, mwIndex *jc)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			tmp += Xt[j] * w[ir[j]];
		}
		value += log(1 + exp(y[i] * tmp));
	}
	value = value / n;
	for (long j = 0; j < d; j++) {
		value += lambda* w[j] * w[j] / 2;
	}
	return value;
}

/// compute_sigmoid_sparse computes the derivative of logistic loss,
/// i.e. ~ y exp(x) / (1 + exp(x)) for sparse data --- sparse x
/// *x - pointer to the first element of the data point
///      (e.g. Xt+jc[i] for i-th example)
/// *w - test point
/// y - label of the training example
/// d - number of nonzeros for current example (*x)
/// *ir - contains row indexes of elements of *x
///		  pointer to the first element of the array 
///		  (e.g. ir+jc[i] for i-th example)
double compute_sigmoid_sparse(double *x, double *w, double y,
	long d, mwIndex *ir)
{
	double tmp = 0;
	// Sparse inner product
	for (long j = 0; j < d; j++) {
		tmp += w[ir[j]] * x[j];
	}
	tmp = exp(y * tmp);
	tmp = y * tmp / (1 + tmp);
	return tmp;
}

/// compute_full_gradient computes the gradient of the entire function,
/// for sparse data matrix. Gradient is changed in place in g. 
/// *Xt - sparse data matrix; examples in columns!
/// *w - test point
/// *y - set of labels
/// *g - gradient; updated in place; input value irrelevant
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
/// For more info about ir, jc convention, see "Sparse Matrices" in 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/matlab-data.html
void compute_full_gradient_sparse(double *Xt, double *w, double *y,
	double *g, long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_sigmoid_sparse(Xt + jc[i], w, 
			y[i], jc[i + 1] - jc[i], ir + jc[i]);
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			g[ir[j]] += Xt[j] * sigmoid;
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
		g[i] += lambda * w[i];
	}
}

//compute_momentum_sparse(temp, w, wold, theta, d, ir, jc + idx);
void compute_momentum_sparse(double *tmp, double *w, double *wold,
	double theta, long d, mwIndex *ir)
{
	for (long j = 0; j < d; j++) {
		tmp[ir[j]] = theta * w[ir[j]] + (1 - theta) * wold[ir[j]];
	}
}

/// Updates the test point *w in place
/// Makes the step only in the nonzero coordinates of *x,
/// and without regularizer. The regularizer step is constant
/// across more iterations --- updated in lazy_updates
/// *x - training example
/// *w - test point; updated in place
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// gold - full gradient computed at point *wold
/// reweight_diag - Reweighted diagonal matrix(vector)
/// d - number of nonzeros of training example *x
/// stepSize - stepsize parameter
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
void update_test_point_sparse(double *x, double *w, double *wold, double *tmp,
	double sigmoid, double sigmoidold,
	double *gold, double * weight, 	double lambda, 
	long d, double stepSize, mwIndex *ir)
{
	for (long j = 0; j < d; j++){
		w[ir[j]] -= stepSize * (x[j] * (sigmoid - sigmoidold) 
			+ lambda* (tmp[ir[j]] - wold[ir[j]]) + weight[ir[j]] * gold[ir[j]]);
	}
}

// summarize the test point
void sum_test_point(double *sumW, double *w, long d, mwIndex *ir) {
	for (long j = 0; j < d; j++)
		sumW[ir[j]] += w[ir[j]];
}

/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For S2GD algorithm
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// i - number of iteration from which this lazy update was called
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// *jc - index of element in data matrix where starts the training
///		  exampls for which the gradient is to be computed
void lazy_update(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) *
			(g[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));
		last_seen[ir[j]] = i;
	}
}

/// Finises the "lazy" updates at the end of outer loop
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// sigma - regularization paramteter
/// iters - number of steps taken in the current outer loop
///			also size of the just finished inner loop
/// d - dimension of the problem
void finish_lazy_updates(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double sigma, long iters, long d)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (iters - last_seen[j]) *
			(g[j] + sigma * (w[j] - wold[j]));
	}
}

/*
USAGE:
hist = svrg_dense(w, Xt, y, lambda, stepsize, iVals, m);
==================================================================
INPUT PARAMETERS:
w (d x 1) - initial point; updated in place
Xt (d x n) - data matrix; transposed (data points are columns); real
y (n x 1) - labels; in {-1,1}
lambda - regularization parameter
stepSize - a step-size
iVals (sum(m) x 1) - sequence of examples to choose, between 0 and (n-1)
m (iters x 1) - sizes of the inner loops
==================================================================
OUTPUT PARAMETERS:
hist = array of function values after each outer loop.
Computed ONLY if explicitely asked for output in MATALB.
*/
mxArray* mig_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize, L;
	long long *iVals, *m;
	double alpha;

	// Other variables
	long long i; // Some loop indexes
	long k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	// Scalar value of the derivative of sigmoid function
	double sigmoid, sigmoidold;
	bool evalf = false; // set to true if function values should be evaluated

	double *wold; // Point in which we compute full gradient
	double *gold; // The full gradient in point wold
	double *hist; // Used to store function value at points in history
	double theta; // Momentum weight
	double *temp; // Temperoy variable
	double *sumW; // Temperoy variable to track the sum of w

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	L = mxGetScalar(prhs[4]); // Smooth parameter of loss function
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	m = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
	alpha = mxGetScalar(prhs[7]);  // parameter need to tuned
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");
	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("m must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[6]); // Number of outer iterations

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	wold = new double[d];
	gold = new double[d];
	temp = new double[d];
	sumW = new double[d];

	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	// Save the snapshot where full gradient was computed
	for (i = 0; i < d; i++) { wold[i] = w[i]; }

	//////////////////////////////////////////////////////////////////
	/// The MiG algorithm ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {
			hist[k] = compute_function_value(wold, Xt, y, lambda, n, d);
		}

		theta = 2 / double(k + 4);
		stepSize = 1.0 / (alpha * theta * L);

		// Initially, compute full gradient at current point w
		compute_full_gradient(Xt, wold, y, gold, n, d, lambda);

		// Initialize the sum of update w
		for (long j = 0; j < d; j++) { sumW[j] = 0; }

		// The inner loop
		for (i = 0; i < m[k]; i++) {
			idx = *(iVals++); // Sample function and move pointer

			// Update the momentum term
			compute_momentum(temp, w, wold, theta, d);

			// Compute current and old scalar sigmoid of the same example
			sigmoid = compute_sigmoid(Xt + d*idx, temp, y[idx], d);
			sigmoidold = compute_sigmoid(Xt + d*idx, wold, y[idx], d);

			update_test_point_dense(Xt + d*idx, w, wold, temp, gold,
				sigmoid, sigmoidold, d, stepSize, lambda);

			for (long j = 0; j < d; j++) { sumW[j] += w[j]; }
		}

		// Save the snapshot
		for (long j = 0; j < d; j++)
		{
			wold[j] = sumW[j] * theta / m[k] + (1 - theta) * wold[j];
		}
	}

	// Evaluate function value if output requested
	if (evalf == true) {
		hist[iters] = compute_function_value(wold, Xt, y, lambda, n, d);
	}


	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////
	delete[] wold;
	delete[] gold;
	delete[] temp;
	delete[] sumW;

	if (evalf == true) { return plhs; }
	else { return 0; }

}

/// svrg_sparse runs the SVRG algorithm for solving L2 regularized 
/// logistic regression on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* mig_sparse(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize, L, alpha;
	long long *iVals, *m;

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	// Scalar value of the derivative of sigmoid function
	double sigmoid, sigmoidold;
	bool evalf = false; // set to true if function values should be evaluated

	double *wold; // Point in which we compute full gradient
	double *gold; // The full gradient in point wold
	//long *last_seen; // used to do lazy "when needed" updates
	double *hist; // Used to store function value at points in history
	double theta; // Momentum weight
	double *temp; // Temperoy variable
	double *sumW; // Temperoy variable to track the sum of w
	double *reweight_diag; //Re-weight matrix

	mwIndex *ir, *jc; // Used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	L = mxGetScalar(prhs[4]); // Smooth parameter of loss function
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	m = (long long*)mxGetPr(prhs[6]); // Sizes of the inner loops
	alpha = mxGetScalar(prhs[7]);  // parameter need to tuned
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");
	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("m must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[6]); // Number of outer iterations
	jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	wold = new double[d];
	gold = new double[d];
	temp = new double[d];
	sumW = new double[d];
	reweight_diag = new double[d];

	// Save the snapshot where full gradient was computed
	for (i = 0; i < d; i++) { wold[i] = w[i]; }
	
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The S2GD algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		// Evaluate function value if output requested
		
		if (evalf == true) {
			hist[k] = compute_function_value_sparse(wold, Xt, y, lambda, n, d, ir, jc);
		}

		theta = 2 / double(k + 4);
		stepSize = 1.0 / (alpha * theta * L);

		// Initially, compute full gradient at snapshot point w.
		compute_full_gradient_sparse(Xt, wold, y, gold, n, d, lambda, ir, jc);

		// Initalized the Re-weight matrix in First epoch
		if (k == 0){
			for (long j = 0; j < d; j++)
				reweight_diag[j] = 1.0 / gold[j];
		}

		// Initialize the sum of update w
		for (long j = 0; j < d; j++) { sumW[j] = 0; }

		// The inner loop
		for (i = 0; i < m[k]; i++) {
			idx = *(iVals++); // Sample function and move pointer

			// Update the momentum term
			/*
			void compute_momentum_sparse(double *tmp, double *w, double *wold,
				double theta, long d, mwIndex *ir)
			*/
			compute_momentum_sparse(temp, w, wold, theta, 
				jc[idx + 1] - jc[idx], ir + jc[idx]);

			// Compute current and old scalar sigmoid of the same example
			sigmoid = compute_sigmoid_sparse(Xt + jc[idx], temp, y[idx],
				jc[idx + 1] - jc[idx], ir + jc[idx]);
			sigmoidold = compute_sigmoid_sparse(Xt + jc[idx], wold, y[idx],
				jc[idx + 1] - jc[idx], ir + jc[idx]);

			// Update the test point
			/*
			void update_test_point_sparse(double *x, double *w, 
			double *wold, double *temp,
			double sigmoid, double sigmoidold,
			double *gold, double * weight, 	double lambda,
			long d, double stepSize, mwIndex *ir)
			*/
			// Update the test point
			update_test_point_sparse(Xt + jc[idx], w, wold, temp,
				sigmoid, sigmoidold, gold, reweight_diag, lambda,
				jc[idx + 1] - jc[idx], stepSize, ir + jc[idx]);

			// void sum_test_point(double *sumW, double *w, long d, mwIndex *ir)
			sum_test_point(sumW, w, jc[idx + 1] - jc[idx], ir + jc[idx]);
		}

		for (long j = 0; j < d; j++)
		{
			wold[j] = sumW[j] * theta / m[k] + (1 - theta) * wold[j];
		}
	}

	// Evaluate the final function value
	if (evalf == true) {
		hist[iters] = compute_function_value_sparse(wold, Xt, y, lambda, n, d, ir, jc);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] wold;
	delete[] gold;
	delete[] temp;
	delete[] sumW;
	delete[] reweight_diag;
	//delete[] last_seen;

	//////////////////////////////////////////////////////////////////
	/// Return value /////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	if (evalf == true) { return plhs; }
	else { return 0; }

}

/// Entry function of MATLAB
/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
/// For more info about this syntax see 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/gateway-routine.html
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// First determine, whether the data matrix is stored in sparse format.
	// If it is, use more efficient algorithm
	if (mxIsSparse(prhs[1])) {
		mexErrMsgTxt("We haven't implement the sparse version here!\nPlease run the Async_Sparse_MiG in ./code/Async_Sparse_MiG/");
		//mexPrintf("We get into sparse version!\n");
		//plhs[0] = mig_sparse(nlhs, prhs);
	}
	else {
		mexPrintf("We get into dense version!\n");
		plhs[0] = mig_dense(nlhs, prhs);
	}
}