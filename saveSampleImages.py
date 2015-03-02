__author__ = 'utsav'
import scipy.io as sio
import Image

mat_contents = sio.loadmat('mnist_all.mat')
i = 0
for key in mat_contents:
    if key.startswith("tr"):
        i = 0
        for arr in mat_contents[key]:
            if i > 20:
                break
            Image.fromarray(arr.reshape(28, 28)).save("sampleImgs/"+key + "-" + str(i) + ".png")
            i += 1

print("done")
