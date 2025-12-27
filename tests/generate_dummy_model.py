
import pickle
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.pnn import PNN

def generate_dummy_model():
    print("Generating dummy PNN model...")
    pnn = PNN()
    
    # 12 features, 2 classes (0: non-fire, 1: fire)
    # Generate 100 samples
    X_train = np.random.rand(100, 12)
    y_train = np.random.randint(0, 2, 100)
    
    pnn.fit(X_train, y_train)
    
    # Save to model_pnn.pkl in root
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_pnn.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pnn, f)
    
    print(f"Dummy model saved to {model_path}")

if __name__ == "__main__":
    generate_dummy_model()
