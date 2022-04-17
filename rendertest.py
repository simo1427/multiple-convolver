import numpy as np
from skimage import io as si
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
import pyopencl as cl
import time
import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

img = np.array([img_as_ubyte(rgb2gray(si.imread("ischdown20x.png"))) for _ in range(1)])

print(img.shape)

# scikit-image yileds kernels that are of different sizes, therefore it is needed they are padded later
mx = -1
kernel5 = []
for theta in np.arange(0, np.pi, np.pi / 8):
    lam = 32
    kernel = gabor_kernel(1 / lam, theta=theta, sigma_x=lam * 0.42, sigma_y=lam * 0.42, n_stds=32 / lam)
    kernel5.append(np.real(kernel))
    kernel5.append(np.imag(kernel))
    mx = max(kernel.shape[0], mx)
# Pad all kernels so that they are of the same size
for i in range(len(kernel5)):
    tmpval=(mx-kernel5[i].shape[0])//2
    kernel5[i]=np.pad(kernel5[i], ((tmpval,),(tmpval,)), mode="constant", constant_values=(0,0))

ksize = kernel5[0].shape[0]

tmpkrn = np.array(kernel5)
print(tmpkrn.shape)

krn = np.zeros((ksize, ksize, 16), dtype=np.float32)
for i in range(ksize):
    for j in range(ksize):
        for k in range(16):
            krn[i, j, k] = tmpkrn[k, i, j]

res = np.zeros((img.shape[0], img.shape[1], img.shape[2], 16), dtype=np.float32)

print(krn.shape)

src_file = open("convolve.cl", "r")
src = src_file.read()
src_file.close()


def platformslist():
    return [platform.name for platform in cl.get_platforms()]


def platformselect(ind):
    return [platform for platform in cl.get_platforms()][ind]


patch_size =77
print(platformslist())
platform = platformselect(0)
devices = platform.get_devices()

context = cl.Context(devices=devices)
prgs_src = cl.Program(context, src)
program = prgs_src.build([f"-DPATCH_SIZE={patch_size}", f"-DKERNEL_VEC_SIZE=16", "-DKERNEL_DATA_TYPE=float"])

img_buff = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
res_buff = cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=res.nbytes)
krn_buff = cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=krn.nbytes)
print(img.nbytes + res.nbytes + krn.nbytes)
queue = cl.CommandQueue(context)
begin = time.perf_counter()
inp = [(img, img_buff), (res, res_buff), (krn, krn_buff)]
out = [(res, res_buff)]
for (arr, buff) in inp:
    cl.enqueue_copy(queue, src=arr, dest=buff)
patchsz = patch_size - ksize
krn_args = [img_buff, res_buff, krn_buff, np.int32(img.shape[1]), np.int32(img.shape[2]), np.int32(ksize)]
print(((img.shape[1] // patchsz + 1), (img.shape[2] // patchsz + 1), img.shape[0]), (ksize, ksize, 1))

completedEvent = program.convolve(queue, ((img.shape[1] // patchsz + 1), (img.shape[2] // patchsz + 1), img.shape[0]),
                                  (1, 1, 1), *krn_args)
completedEvent.wait()
for (arr, buff) in out:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()
print(f"GPU convolution of {img.shape[0]} images:{time.perf_counter() - begin}")
print(res.shape)
for iimg in range(img.shape[0]):
    for i in range(16):
        si.imsave(f"output-{iimg}-filter-{i}-mod.tif", res[iimg, :, :, i])
print(time.perf_counter() - begin)
