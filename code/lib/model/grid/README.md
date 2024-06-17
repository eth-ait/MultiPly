# Cuda-based grid sample implimentation with second order derivative support.

For the forward and backward pass it uses the pytorch version of grid sample.
Only the second backward is implimented here.

**Tested only with Pytorch 10.1 and 10.2**

**Reflection padding is not supported**

Usage:
```
import cuda_gridsample as cu

cu.grid_sample_2d(image, optical, padding_mode='border', align_corners=True)
cu.grid_sample_3d(volume, optical, padding_mode='border', align_corners=True)

```
