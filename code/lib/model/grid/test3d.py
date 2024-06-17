import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import cuda_gridsample as cu
import naive_gridsample as nv
from torch.autograd import grad
from functools import partial

import unittest
from torch.testing import assert_close

class CudaGridsampleTest(unittest.TestCase):
           
    def test_naive_constant(self):
        image = np.arange(27).reshape(1, 1, 3, 3, 3)
        optical = np.array([0.1, 0.1, 0.1]).reshape(1, 1, 1, 1, 3)
        self.cmp_with_naive(image, optical)
     
    def test_naive_oob(self):
        image = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).reshape(1, 1, 2, 2, 2)
        optical = np.array([0.1, 1.1, 0.1]).reshape(1, 1, 1, 1, 3)
        self.cmp_with_naive(image, optical)
 
    def test_naive_random(self):
        for i in range(10):
            input, grid = self.create_random_input(oob=False)
            self.cmp_with_naive(input, grid)  

    def test_gradcheck_constant(self):
        image = np.arange(27).reshape(1, 1, 3, 3, 3)
        optical = np.array([-2.1, 0.1, 0.1]).reshape(1, 1, 1, 1, 3)
        self.gradcheck(image, optical, padding_mode='zeros')
    
    def test_gradcheck_random(self):
        for i in range(10):
            input, grid = self.create_random_input(max_dim=5)
            self.gradcheck(input, grid)

    def test_gradcheck_random_oob_zeros(self):
        for i in range(10):
            input, grid = self.create_random_input(oob=True, max_dim=5)
            self.gradcheck(input, grid, padding_mode='zeros')

    def test_gradcheck_random_oob_border(self):
        for i in range(10):
            input, grid = self.create_random_input(oob=True, max_dim=5)
            self.gradcheck(input, grid, padding_mode='border')

    def test_grad_output(self):
        for i in range(10):
            input, grid, grad_output = self.create_random_input(oob=True, max_dim=5, with_grad_output=True)

            input = torch.DoubleTensor(input)
            grid = torch.DoubleTensor(grid)
            grad_output = torch.DoubleTensor(grad_output)

            input = input.cuda()
            grid = grid.cuda()
            grad_output = grad_output.cuda()

            input.requires_grad = True
            grid.requires_grad = True
            grad_output.requires_grad = True
            torch.autograd.gradcheck(lambda grad_output, input, grid: cu._GridSample3dBackward.apply(grad_output, input, grid), (grad_output, input, grid))
    
    def test_gradcheck_random_nocorners(self):
        for i in range(10):
            input, grid = self.create_random_input(oob=True, max_dim=5)
            self.gradcheck(input, grid, padding_mode='zeros', align_corners=False)

    def test_float(self):
        image = np.arange(27).reshape(1, 1, 3, 3, 3)
        optical = np.array([0.1, 0.1, 0.1]).reshape(1, 1, 1, 1, 3)
        self.cmp_with_naive(image, optical, double=False) 

    
    def test_strided(self):
        for i in range(100):
            input, grid = self.create_random_input(max_dim=5)
            if i % 2 == 0:
                bs_stride = np.random.randint(1, input.shape[0] + 1) 
                c_stride = np.random.randint(1, input.shape[1] + 1)
                d_stride = np.random.randint(1, input.shape[2] + 1) 
                h_stride = np.random.randint(1, input.shape[3] + 1) 
                w_stride = np.random.randint(1, input.shape[4] + 1)

                dg_stride = np.random.randint(1, grid.shape[1] + 1)               
                hg_stride = np.random.randint(1, grid.shape[2] + 1)
                wg_stride = np.random.randint(1, grid.shape[3] + 1)
            else:
                bs_stride = np.random.randint(1, 3) 
                c_stride = np.random.randint(1, 3)
                d_stride = np.random.randint(1, 3)  
                h_stride = np.random.randint(1, 3) 
                w_stride = np.random.randint(1, 3)

                dg_stride = np.random.randint(1, 3) 
                hg_stride = np.random.randint(1, 3)
                wg_stride = np.random.randint(1, 3)


            input = torch.DoubleTensor(input)
            grid = torch.DoubleTensor(grid)
      
            input.requires_grad = True
            grid.requires_grad = True

            image = input.cuda()
            optical = grid.cuda()
           
            image = image[::bs_stride, ::c_stride, ::d_stride, ::h_stride, ::w_stride]
            optical = optical[::bs_stride, ::hg_stride, ::dg_stride, ::wg_stride]

            self.assertTrue(torch.autograd.gradcheck(partial(cu.grid_sample_3d), inputs=(image, optical)))
            self.assertTrue(torch.autograd.gradgradcheck(partial(cu.grid_sample_3d), inputs=(image, optical)))

    
    def test_use_case(self):
        torch.set_default_dtype(torch.float64)
        for i in range(10):
            input, grid = self.create_random_input(max_dim=10)

            l1 = nn.Conv3d(input.shape[1], input.shape[1], 1)
            l2 = nn.Conv3d(input.shape[1], 1, 1)

            image = torch.DoubleTensor(input)
            optical = torch.DoubleTensor(grid)
      
            image.requires_grad = True
            optical.requires_grad = True

            image = image.cuda()
            optical = optical.cuda()
            l1.cuda()
            l2.cuda()

            def fn(image, optical):
                out = l1(image)
                out = F.relu(out)
                out = cu.grid_sample_3d(out, optical)
                out = l2(out)
                out = out * out
                out = out.sum()
                return out

            self.assertTrue(torch.autograd.gradcheck(partial(fn), inputs=(image, optical), nondet_tol=1e-05))
            self.assertTrue(torch.autograd.gradgradcheck(partial(fn), inputs=(image, optical), nondet_tol=1e-05))
        torch.set_default_dtype(torch.float32)
    

    def create_random_input(self, oob=False, max_dim=20, with_grad_output=False):
        bs = np.random.randint(1, max_dim)
        c = np.random.randint(1, max_dim)
        d = np.random.randint(1, max_dim)
        h = np.random.randint(1, max_dim)
        w = np.random.randint(1, max_dim)

        dg = np.random.randint(1, max_dim) 
        hg = np.random.randint(1, max_dim)
        wg = np.random.randint(1, max_dim)
        if oob:
            bounds = (-2, 2)
        else:
            bounds = (-1, 1)
        grid = np.random.uniform(bounds[0], bounds[1], size=(bs, dg, hg, wg, 3))
        input = np.random.normal(size=(bs, c, d, h, w))
        if not with_grad_output:
            return input, grid
        else:
            grad_output = np.random.normal(size=(bs, c, dg, hg, wg))
            return input, grid, grad_output

    def cmp_with_naive(self, image, optical, double=True):
        if double:
            image = torch.DoubleTensor(image)
            optical = torch.DoubleTensor(optical)
        else:
            image = torch.FloatTensor(image)
            optical = torch.FloatTensor(optical)
 
        image.requires_grad = True
        optical.requires_grad = True
        
        nv_out = nv.grid_sample_3d(image, optical)
        nv_out = torch.sum(nv_out ** 2)

        nv_grad_image, nv_grad_optical = grad(nv_out, [image, optical], create_graph=True)
        nv_grad2_image, nv_grad2_optical = grad(torch.sum(nv_grad_image) + torch.sum(nv_grad_optical), [image, optical])
        
        image = image.cuda()
        optical = optical.cuda()

        out = cu.grid_sample_3d(image, optical, padding_mode='border', align_corners=True)
        out = torch.sum(out ** 2)
        grad_image, grad_optical = grad(out, [image, optical], create_graph=True)
        grad2_image, grad2_optical = grad(torch.sum(grad_image) + torch.sum(grad_optical), [image, optical])
          
        assert_close(nv_out, out.cpu())
 
        assert_close(nv_grad_image, grad_image.cpu())
        assert_close(nv_grad_optical, grad_optical.cpu())
        assert_close(nv_grad2_image, grad2_image.cpu())

        assert_close(nv_grad2_optical, grad2_optical.cpu())
        assert_close(nv_grad2_image, grad2_image.cpu())

    def gradcheck(self, image, optical, padding_mode='border', align_corners=True):
        image = torch.DoubleTensor(image)
        optical = torch.DoubleTensor(optical)
      
        image.requires_grad = True
        optical.requires_grad = True

        image = image.cuda()
        optical = optical.cuda()

        self.assertTrue(torch.autograd.gradcheck(partial(cu.grid_sample_3d, padding_mode=padding_mode, align_corners=align_corners), inputs=(image, optical)))
        self.assertTrue(torch.autograd.gradgradcheck(partial(cu.grid_sample_3d, padding_mode=padding_mode, align_corners=align_corners), inputs=(image, optical)))


if __name__ == "__main__":
    unittest.main()

