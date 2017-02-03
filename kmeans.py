import numpy as np
import random
import sys, getopt

file_path = 'C:\\Users\\Jacqueline\\Desktop\\ML\\test1'

def convert_file( file_path ):
    '''
    ===== DESCRIPTION =================================
    Helper function to load data from a given file
    into the environment as a matrix.
        
    ===== INPUT =======================================
    file_path - (String) Path to file with (N+1) lines:
        Line 1: d (dimension); N (# data points).
        Lines 2-(N+1): N d-dimensional data points.
    
    ===== OUTPUT ======================================
    data - (N x d) matrix containing N data points,
    each of dimension d (each row is a datum).
    '''
    
    with open(file_path, 'r') as file:
        data = np.matrix(np.loadtxt(file, dtype = "int", skiprows = 1))
        del file, file_path
        return data
    print 'File not found.'
    return None


def find_centers( data, k, classes = None):
    '''
    ===== DESCRIPTION ========================
    Helper function to find k distinct centers
    among given data points, depending on the
    presence/absence of classes argument.
    
    ===== INPUT ==============================
    data - (N x d) matrix containing N data
        points, each of dimension d.
    k - (int > 0) # of centers to be found.
    classes - (Optional) N-dimensional vector
        dictating how to compute the centers:
        Not None (Provided): Compute averages
            of data in clusters (classes)
        None: Randomly select k distinct
            data points among given data.
    
    ===== OUTPUT =============================
    centers - (k x d) matrix containing the k
    centers chosen for the k clusters.
    '''
    
    # To be used only once in the k-means algorithm
    if classes is None:             
        # Ensure centers have unique rows (data points)
        temp = np.matrix(np.vstack({tuple(row) for row in np.asarray(data)}))
        sample = np.random.choice(xrange(temp.shape[0]), k, replace = False)     
        centers = temp[sample, :]
        
    # To be used for all following iterations of the k-means algorithm
    # Recalculate centers as averages of nearest neighbors
    else:                           
        d = data.shape[1]
        centers = np.matrix(np.zeros((k, d)))
        
        for i in xrange(k):
            # Get data currently in class i
            ids = np.where(classes == i)[0]
            temp = data[ids, :]      
            
            # Add rows (data points) for class i & divide by class size i
            temp = np.matrix(np.asarray([ sum(temp[:, j]) for j in xrange(d) ]))
            temp /= np.size(ids)
            centers[i] = temp                   # Store class data in centers[i]
            del temp, ids
    return centers


def compute_distances( data, centers ):
    '''
    ===== DESCRIPTION ========================
    Helper function to compute distances 
    between (N) data points & (k) centers.
    
    ===== INPUT ==============================
    data - (N x d) matrix containing N data
        points, each of dimension d.
    centers - (k x d) matrix containing k
        chosen centers, each of dimension d.
    
    ===== OUTPUT =============================
    D - (N x k) matrix of squared 2-normed 
    distances between (N) data points
    & (k) chosen centers.
    '''
    
    N, d = data.shape
    k = centers.shape[0]

    '''
    D(i,j) = ||data(i) - centers(j)||^2         <- Squared L2-norms
    = ||data(i)||^2 + ||centers(j)||^2 - 2*data(i)*transpose(centers(j)).
    
    Rather than looping over each (i,j) pair, we create D = a + b - 2c, where:
	a = (N x k) matrix of k (identical) N-dimensional column vectors 
	(entries are row sums of the matrix whose entries are the squared entries of data);
	b = (N x k) matrix of N (identical) k-dimensional row vectors 
	(entries are row sums of the matrix whose entries are the squared entries of centers);
	c = (N x k) matrix given by (data * transpose(centers)).
    '''
    D = ( np.tile(np.square(data) * np.mat(np.ones(d)).T, k)
        + np.tile(np.square(centers) * np.mat(np.ones(d)).T, N).T
        - 2 * data.dot(centers.T) )
    
    return D


def kmeans( k, data ):
    '''
    ===== DESCRIPTION =============================
    Main k-means algorithm for finding k clusters
    and their centers in a given set of data.
    
    ===== INPUT ===================================
    k - (int > 0) # of clusters to be found
    data - (N x d) matrix containing N data
        points, each of dimension d.
    
    ===== OUTPUT ==================================
    centers - (k x d) matrix containing the k
        centers chosen for the k clusters.
    classes - (N x 2) matrix of class assignments
        and corresponding distances to centers.
    class_sizes = (k x 1) vector of class sizes.
    error - SSE of data and corresponding centers.
    counter - (int) # iterations used in method.
    '''
    
    N = data.shape[0]
    centers = find_centers(data, k)
    D = compute_distances(data, centers)
    
    # Class assignments vector with corresponding distances
    classes = np.matrix(np.zeros((N, 2)))
    classes[:, 0] = np.nanargmin(D, axis = 1)           # Class assignments
    classes[:, 1] = np.nanmin(D, axis = 1)              # Min. dist. for each datum
    
    # Guard against empty clusters
    while np.unique(np.asarray(classes[:, 0])).size < k:
        for i in xrange(k):
            if (i in classes[:, 0]) == False:
                # Replace empty cluster center with some point that is
                # some random perturbation of the center of the largest cluster
                (values, counts) = np.unique(np.asarray(classes[:, 0]), return_counts = True)
                id = np.argmax(counts)
                centers[i] = centers[id] + np.random.random_sample()
                
                # Given new center, recompute distances & populate classes
                D = compute_distances(data, centers)
                classes[:, 0] = np.nanargmin(D, axis = 1)
                classes[:, 1] = np.nanmin(D, axis = 1)
                i = 0                                   # Restart check for empty clusters

    # Initialize old and current errors
    error = np.sum(np.square(classes[:, 1]))            # Current SSE (trial 1)
    old_error = sys.maxint
    
    # Iterate finding nearest neighbors of the k centers
    counter = 1
    while abs((error - old_error) / old_error) >= 0.01:         # Stop if <1% change in the SSE
        # Find the k centers and compute distances
        centers = find_centers(data, k, classes[:, 0])
        D = compute_distances(data, centers)
        
        # Populate classes
        classes[:, 0] = np.nanargmin(D, axis = 1)       # Class assignments
        classes[:, 1] = np.nanmin(D, axis = 1)          # Distances to paired centers

        while np.unique(np.asarray(classes[:, 0])).size < k:
            # If a cluster becomes empty, replace that center with a
            # small random perturbation of the center of the largest cluster.
            for i in xrange(k):
                if (i in classes[:, 0]) == False:
                    # Find largest cluster center; adjust current empty cluster center
                    (values, counts) = np.unique(np.asarray(classes[:, 0]), return_counts = True)
                    id = np.argmax(counts)
                    centers[i] = centers[id] + np.random.random_sample()
                    
                    # Recompute distances, repopulate classes, and reset index
                    D = compute_distances(data, centers)
                    classes[:, 0] = np.nanargmin(D, axis = 1)
                    classes[:, 1] = np.nanmin(D, axis = 1)
                    i = 0
        counter += 1
        temp = error                                    # Temporarily save last iteration's SSE
        error = np.sum(np.square(classes[:, 1]))        # Current iteration SSE (overwrite error)
        old_error = temp                                # Overwrite old_error
        
    # Find the cluster sizes for the 'converged' clusters m_i, for i = 0, ..., k-1
    (values, counts) = np.unique(np.asarray(classes[:, 0]), return_counts = True)
    class_sizes = np.matrix(np.zeros(k))
    class_sizes = counts
    print 'Centers', centers
    return centers, classes, class_sizes, error, counter


def test_k( file_path, lim ):
    '''
    ===== DESCRIPTION =================================
    Main test function for implementing the k-means
    algorithm for varying k = 1, ... , N/2.
    
    ===== INPUT =======================================
    file_path - (String) Path to file with (N+1) lines:
        Line 1: d (dimension); N (# data points).
        Lines 2-(N+1): N d-dimensional data points.
    lim - (int > 0) Desired maximum value of k
    
    ===== OUTPUT ======================================
    data - (N x d) matrix containing N data points,
        each of dimension d (each row is a datum).
    best_k - (int) Optimal k value (lowest error).
    best_error - (float) Optimal error value.
    '''
    
    data = convert_file( file_path )
    N, d = data.shape
    # lim = int(N / 2)
    
    # Initialize error trackers for given k value (7 trials each)
    error_compare = np.matrix(np.zeros((lim, 7)))
    min_errors = np.matrix(np.zeros((lim, 1)))           # Best error per k (across 7 trials)

    # For each choice of k, run the k-means algorithm 7 times.
    for k in range(1, lim + 1):
        centers_k = np.matrix(np.zeros((k, d)))
        for j in xrange(7):
            temp = kmeans(k, data)
            # Error of current run (out of 7 runs) for the given value of k
            error = temp[3]
            error_compare[k - 1, j] = error
            
            if j == 0:
                centers_k = temp[0]
            else:   # Overwrite centers if current error is better than previous error
                if error < error_compare[k - 1, j - 1]:
                    centers_k = temp[0]
        # Find the best run for the given value of k
        min_errors[k - 1] = np.nanmin(error_compare[k - 1])
        
    # Minimum error for all chosen values of k
    best_k = np.nanargmin(min_errors) + 1
    best_error = np.nanmin(min_errors)
    return min_errors, error_compare, best_k, best_error, centers_k
    
def main(argv):
    inputfile = ''
    k_max = 0
    try:
        opts, args = getopt.getopt(argv, "hi:k:")
    except getopt.GetoptError:
        print 'kmeans.py -i <inputfile> -k <max k>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'kmeans.py -i <inputfile> -k <max k>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-k"):
            k_max = int(arg)
        
    test_k(inputfile, k_max)
    return

if __name__ == "__main__":
   main(sys.argv[1:])
