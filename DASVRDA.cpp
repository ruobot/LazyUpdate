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

// compute_momentum(temp, w, z, theta, d);
void compute_momentum(double *tmp, double *w, double *z,
	double theta, long d) {
	for (long j = 0; j < d; j++)
	{
		tmp[j] = (1 - theta) * w[j] + theta * z[j];
	}
}

//update_avg_grad(Xt + d*idx, temp, wold, gold,
//  	sigmoid, sigmoidold, d, lambda, prob[idx], theta);
void update_avg_grad(double *x, double *temp, double *wold, 
	double *gold, double *avg,
	double sigmoid, double sigmoidold, long d,
	double lambda, double p, double theta)
{
	for (long j = 0; j < d; j++)
	{
		double g = gold[j] + (x[j] * (sigmoid - sigmoidold)
			+ lambda * (temp[j] - wold[j])) * p;
		avg[j] = (1 - theta) * avg[j] + theta * g;
	}
}

// update_test_point_dasvrda(z, z0, avg, stepSize, theta, theta_old);
void update_test_point_dasvrda(double *z, double *z0, double *avg,
	double stepSize, double theta, double theta_old, long d)
{
	for (long j = 0; j < d; j++)
	{
		z[j] = z0[j] - stepSize * avg[j] / (theta * theta_old);
	}
}

/*
USAGE:
hist = sdamm_dense(w, Xt, y, lambda, stepsize, iVals, m);
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
mxArray* dasvrda_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y, *prob;
	double lambda, stepSize, L;
	long long *iVals, *m;	
	//double alpha;

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
	//double theta; // Momentum weight
	double *temp; // Temperoy variable
	double *avg; // Temperoy variable to track the sum of w

	//////////////////////////////////////////////////////////////////
	/// outer momentum terms
	//////////////////////////////////////////////////////////////////
	double *z;
	double *wold_;
	double *z0;

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	L = mxGetScalar(prhs[4]); // Smooth parameter of loss function
	prob = mxGetPr(prhs[5]); // probability of index for sampling
	iVals = (long long*)mxGetPr(prhs[6]); // Sampled indexes (sampled in advance)
	m = (long long*)mxGetPr(prhs[7]); // Sizes of the inner loops
	//alpha = mxGetScalar(prhs[8]);  // parameter need to tuned
	double gamma = ( 3.0 + sqrt(9.0 + 8.0 / (double(m[0]) + 1)) ) / 2;
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("iVals must be int64");
	if (!mxIsClass(prhs[7], "int64"))
		mexErrMsgTxt("m must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[7]); // Number of outer iterations

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	wold = new double[d];
	gold = new double[d];
	temp = new double[d];
	avg = new double[d];
	z = new double[d];
	wold_ = new double[d];
	z0 = new double[d];

	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	// Save the snapshot where full gradient was computed
	for (i = 0; i < d; i++) { 
		wold[i] = w[i]; 
		z[i] = w[i];
		wold_[i] = w[i];
	}

	stepSize = 1 / ((1 + gamma * (m[0] + 1)) * L);
	 
	//////////////////////////////////////////////////////////////////
	/// The DASVRDA algorithm ////////////////////////////////////////
	//////////////////////////////////////////////////////////////////
	double thetas, thetas_old;
	double weight1, weight2;

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {
			hist[k] = compute_function_value(wold, Xt, y, lambda, n, d);
		}

		thetas = (1 - 1.0 / gamma) * (k + 3) / 2;
		thetas_old = (1 - 1.0 / gamma) * (k + 2) / 2;

		weight1 = (thetas_old - 1) / thetas;
		weight2 = thetas_old / thetas;

		for (long j = 0; j < d; j++)
		{
			w[j] = wold[j] + weight1 * (wold[j] - wold_[j]) 
				+ weight2 * (z[j] - wold[j]);
		}

		// Initially, compute full gradient at current point w
		compute_full_gradient(Xt, wold, y, gold, n, d, lambda);

		// Initialize the sum of update w
		for (long j = 0; j < d; j++) { 
			avg[j] = 0; 
			z[j] = w[i];
			z0[j] = w[j];
		}

		// The inner loop
		for (i = 0; i < m[k]; i++) {
			idx = *(iVals++); // Sample function and move pointer

			double theta = 2.0 / (i + 2);   // here we use the reciprocal in paper
			double theta_old = 2.0 / (i + 1);  //....

			// Update the momentum term
			compute_momentum(temp, w, z, theta, d);

			// Compute current and old scalar sigmoid of the same example
			sigmoid = compute_sigmoid(Xt + d*idx, temp, y[idx], d);
			sigmoidold = compute_sigmoid(Xt + d*idx, wold, y[idx], d);

			update_avg_grad(Xt + d*idx, temp, wold, gold, avg,
				sigmoid, sigmoidold, d, lambda, prob[idx], theta);

			update_test_point_dasvrda(z, z0, avg, stepSize, theta, theta_old, d);

			//compute_momentum(w, w, z, theta, d);
			for (long j = 0; j < d; j++)
			{
				w[j] = (1 - theta) * w[j] + theta * z[j];
			}
		}

		for (long j = 0; j < d; j++)
		{
			wold_[j] = wold[j];
			wold[j] = w[j];
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
	delete[] avg;
	delete[] wold_;
	delete[] z;
	delete[] z0;

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
		//plhs[0] = dasvrda_sparse(nlhs, prhs);
	}
	else {
		mexPrintf("We get into dense version!\n");
		plhs[0] = dasvrda_dense(nlhs, prhs);
	}
}