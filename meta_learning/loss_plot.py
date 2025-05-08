import numpy as np
import matplotlib.pyplot as plt

dir_ = "log/meta_mlpt_h4_bl15_f0"
loss_seq = np.load(dir_+"/loss_record.npy")
print("epoch: ", loss_seq.shape[0], "final_loss: ", loss_seq[-1])
plt.plot(loss_seq)
plt.grid()
plt.show()