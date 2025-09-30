import numpy as np

def interpolate_data(data, valid_indices, total_len):
    aligned_data = np.zeros((total_len, data.shape[1]))

    # Fill in known values
    for i, idx in enumerate(valid_indices):
        aligned_data[idx] = data[i]

    # Get the list of missing indices
    all_indices = np.arange(total_len)
    missing_indices = np.array([i for i in all_indices if i not in valid_indices])

    # Perform linear interpolation
    if len(valid_indices) > 1:
        for i in range(len(missing_indices)):
            idx = missing_indices[i]
            # Find the nearest valid indices before and after the missing index
            prev_idx = max([i for i in valid_indices if i < idx], default=None)
            next_idx = min([i for i in valid_indices if i > idx], default=None)
            
            if prev_idx is not None and next_idx is not None:
                # Interpolate based on the surrounding values
                weight = (idx - prev_idx) / (next_idx - prev_idx)
                aligned_data[idx] = aligned_data[prev_idx] * (1 - weight) + aligned_data[next_idx] * weight
            elif prev_idx is not None:
                # Extrapolate using the previous valid value
                aligned_data[idx] = aligned_data[prev_idx]
            elif next_idx is not None:
                # Extrapolate using the next valid value
                aligned_data[idx] = aligned_data[next_idx]

    return aligned_data