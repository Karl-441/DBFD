import numpy as np
from scipy.spatial.distance import cdist

class PNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.sigmas = {} # Class -> Sigma
        self.classes = []
        
    def fit(self, X, y):
        """
        Stores training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes = np.unique(y)
        # Initialize sigmas
        for c in self.classes:
            self.sigmas[c] = 1.0

    def optimize_ecm(self):
        """
        Optimizes smoothing parameters (sigmas) using an ECM-like iterative approach.
        Since we don't have the exact closed-form Q-function from the user's specific reference,
        we implement a coordinate descent (ECM is a generalized EM/coordinate descent) 
        maximizing the Leave-One-Out accuracy/likelihood.
        We optimize sigma for each class sequentially.
        """
        print("Starting ECM optimization for PNN smoothing factors...")
        
        # Iteratively optimize each sigma
        max_iter = 5
        for it in range(max_iter):
            changes = 0
            for c in self.classes:
                best_s = self.sigmas[c]
                best_score = self.evaluate_loo()
                
                # Line search for this class's sigma
                search_space = np.linspace(0.1, 3.0, 15)
                
                current_best_s = best_s
                
                for s in search_space:
                    self.sigmas[c] = s
                    score = self.evaluate_loo()
                    if score > best_score:
                        best_score = score
                        current_best_s = s
                
                if abs(current_best_s - best_s) > 1e-3:
                    changes += 1
                
                self.sigmas[c] = current_best_s
            
            print(f"Iteration {it+1}: Sigmas = {self.sigmas}, Score = {best_score:.4f}")
            if changes == 0:
                break
                
    def evaluate_loo(self):
        """
        Leave-One-Out Cross Validation accuracy.
        """
        n = len(self.X_train)
        correct = 0
        
        # Precompute sq euclidean distances: (N, N)
        dists = cdist(self.X_train, self.X_train, 'sqeuclidean')
        
        for i in range(n):
            true_cls = self.y_train[i]
            scores = {}
            
            for c in self.classes:
                sigma = self.sigmas[c]
                # Indices of samples in class c
                c_indices = np.where(self.y_train == c)[0]
                # Exclude self (LOO)
                c_indices = c_indices[c_indices != i]
                
                if len(c_indices) == 0:
                    scores[c] = 0
                    continue
                
                # Distances from sample i to all training samples of class c
                d = dists[i, c_indices]
                
                # Kernel: exp( -dist / 2sigma^2 )
                # Note: Denominator (2*pi*sigma^2)^(d/2) is constant for comparison if sigma is same,
                # but here sigma varies per class, so we MUST include the normalization factor.
                # Dim = 12
                dim = self.X_train.shape[1]
                norm = (2 * np.pi * sigma**2) ** (dim / 2)
                
                kernels = np.exp(-d / (2 * sigma**2))
                # Parzen window density estimate: Sum(kernels) / (N * norm)
                density = np.sum(kernels) / (len(c_indices) * norm)
                scores[c] = density
            
            # Predict
            if not scores:
                pred = self.classes[0]
            else:
                pred = max(scores, key=scores.get)
            
            if pred == true_cls:
                correct += 1
                
        return correct / n

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X[None, :]
            
        n_test = len(X)
        dists = cdist(X, self.X_train, 'sqeuclidean')
        
        preds = []
        dim = self.X_train.shape[1]
        
        for i in range(n_test):
            scores = {}
            for c in self.classes:
                sigma = self.sigmas[c]
                c_indices = np.where(self.y_train == c)[0]
                
                if len(c_indices) == 0:
                    scores[c] = 0
                    continue
                
                d = dists[i, c_indices]
                norm = (2 * np.pi * sigma**2) ** (dim / 2)
                kernels = np.exp(-d / (2 * sigma**2))
                density = np.sum(kernels) / (len(c_indices) * norm)
                scores[c] = density
            
            pred = max(scores, key=scores.get)
            preds.append(pred)
            
        return np.array(preds)
