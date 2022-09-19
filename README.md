# multiple-convolver
**Work in progress**
This is the interim name of what is to become a tool for convolving a batch of images using more than one convolution filter (i.e., filter bank).

The tool is split in two parts - host-side and device-side.

The device-side code is written in OpenCL which gives one the freedom of deivce for execution - a CPU, GPU, FPGA, etc.

The host-side code is as of now implemented in Python and uses [`pyopencl`](https://github.com/inducer/pyopencl)
The current demo provided here also requires [`scikit-image`](https://github.com/scikit-image/scikit-image)

## Running instructions

1. Set up an OpenCL runtime. This is dependent on the vendor of the device on which code is to be executed, follow their instructions
2. Install `pyopencl` and `scikit-image`
    ```pip install pyopencl scikit-image```
3. Setting up the execution
   1. Choose a valid OpenCL platform.
   2. The `img` array in `rendertest.py` should contain grayscale images **that have exactly the same shape**.
   3. Have an array with filter kernels in `krn`. An example with 16 Gabor filter kernels is provided. **All filter kernels must have exactly the same shape**
   4. In `rendertest.py` the last part of the code has to do with saving the images. Change the filename but ensure that for each filter and for each image there are different filenames.