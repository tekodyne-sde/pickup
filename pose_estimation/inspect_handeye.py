import numpy as np

data = np.load("handeye_result.npz")
print("Keys in handeye_result.npz:", data.files)
for k in data.files:
    arr = data[k]
    print(f"\n{k}: shape={arr.shape if hasattr(arr, 'shape') else 'scalar'}")
    print(arr)