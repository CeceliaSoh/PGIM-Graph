import numpy as np

# Step 1: load numpy file
arr = np.load("dataset/feature_ccr_v3.1.1_within_250m_train.npy")   # shape: [N, T, D]
# output_path = "dataset/feature_ccr_v3.1.1_within_250m_test.npy"
print("Shape:", arr.shape)

# Step 2: find columns (features) with any value > 10
mask = arr > 1000                    # [N, T, D]
feature_has_large = mask.any(axis=(0,1)) # [D]

feature_indices = np.where(feature_has_large)[0]

print("Feature indices with values > 10:")
print(feature_indices)
# print(arr[100,201,feature_indices])  # print values of those features for the first sample and time step


# # Step 2: transform feature index 7
# arr[:, :, 7] = np.clip(arr[:, :, 7] / 10.0, None, 10)

# # Step 3: save to new file
# np.save(output_path, arr)

# print(f"Saved processed file to: {output_path}")

print("Mean and std of feature -2:")
print(np.mean(arr[:,:,-2]), np.std(arr[:,:,-2]), np.min(arr[:,:,-2]), np.max(arr[:,:,-2]))