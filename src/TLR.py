import numpy as np
from sklearn.linear_model import LinearRegression

# Example data
y = np.array([3, 2, 5, 5, 6, 5, 7, 8, 7, 10, 9])

def latest_monotonic_sequence(sequence):
    """ Finds the latest monotonically increasing or decreasing subsequence in a 1D array """
    n = len(sequence)
    if n == 0:
        return np.array([]), np.array([])
    
    # Start from the last element and look backwards
    last_idx = n - 1
    indices = [last_idx]
    values = [sequence[last_idx]]
    
    # Determine if the sequence is increasing or decreasing
    is_increasing = sequence[last_idx - 1] < sequence[last_idx] if last_idx > 0 else True
    
    for i in range(last_idx - 1, -1, -1):
        if (is_increasing and sequence[i] <= sequence[i + 1]) or \
           (not is_increasing and sequence[i] >= sequence[i + 1]):
            indices.append(i)
            values.append(sequence[i])
        else:
            break
    
    # Reverse to make the sequence and indices in ascending order
    indices.reverse()
    values.reverse()
    
    return np.array(indices), np.array(values)

def truncate_linear_regression(history_sequence, future_steps):
    # Extract the latest monotonically increasing subsequence
    indices, values = latest_monotonic_sequence(history_sequence)

    # Reshape indices for sklearn
    indices = np.array(indices).reshape(-1, 1)

    # Create and train the model
    model = LinearRegression()
    model.fit(indices, values)

    # Number of future steps to predict
    # future_steps = 3
    # Create future indices array from the last index of monotonically increasing sequence
    future_indices = np.arange(indices[-1, 0] + 1, indices[-1, 0] + 1 + future_steps).reshape(-1, 1)

    # Predict future values
    future_values = model.predict(future_indices)
    print(f"Future values for the next {future_steps} steps: {future_values}")
