import numpy as np
import torch
import cv2

points1 = np.float32([[30, 30], [10, 40], [40, 10], [5, 15]]).T
points2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]]).T

a = np.arange(9.).reshape([3, 3])
print(a, '\n', a[:, 2])
for i in range(4):
    print(i)

#x = torch.from_numpy(np.arange(12.).reshape([4, 3])).requires_grad_(True)
x = np.arange(9.).reshape([3, 3])
print(x)
print(x[:, 1])
# print(x.max(1)[0].unsqueeze(-1).size())
#z = y * y * 3
#out = z.mean()
# print(z, '\nout=', out)
#out.backward()
# print('x.grad={}'.format(x.grad))