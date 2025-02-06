import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Regular_RMSES = np.load("training-rmses-augment=False.npy")
Rotated_RMSES = np.load("training-rmses-augment=True.npy")

plt.figure(1)
plt.plot(range(1,100000),Rotated_RMSES, label='RMSE for Rotated Data Set')
plt.title('RMSE for Rotated Data Set')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show(block=False)  # Use non-blocking show

plt.figure(2)
plt.plot(range(1,100000),Regular_RMSES, label='RMSE for Regular Data Set')
plt.title('RMSE for Regular Data Set')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show(block=False)  # Use non-blocking show
plt.show()