import torch

from torch.distributions.multivariate_normal import MultivariateNormal
import einops
#from gpytorch.distributions import MultivariateNormal
def sample_from_batch_multivariate_normal(cov_matrix, c=2,h=1,w=1,batch_size=128, aug_dim=6, eps=1e-5,device='cpu'):

    # Ensure covariance matrix has shape [batch_size, dim, dim]
    assert cov_matrix.shape == (batch_size, aug_dim, aug_dim), "Covariance matrix must have shape [batch_size, dim, dim]"

    # Zero mean for each distribution in the batch
    mean = torch.zeros(batch_size, aug_dim,device=device)
  #  I_eps = torch.eye(aug_dim,aug_dim)[None,:,:]*eps
  #  I_eps[:,0,0] = 0.0
   # cov_matrix += I_eps

  #  mean = mean.type(torch.float64)
  #  cov_matrix = cov_matrix.type(torch.float64)
    # Create the batch of Multivariate Normal distributions
    mvn = MultivariateNormal(mean, covariance_matrix=cov_matrix)

    # Sample from the distribution
    n_samples = int(c*h*w)
    samples = mvn.sample(sample_shape=(n_samples,))  # Samples will have shape [n_samples, batch_size, dim]
    samples = einops.rearrange(samples, '(C H W) B K -> B C H W K', C=c, H=h, W=w)
    #print(f'SAMPLES BEFORE {samples.shape}')
    ##  samples = samples.type(torch.float32)
    #samples = samples.permute(1,0,2)
    #print(f'SAMPLES AFTER {samples.shape}')
    #samples = samples[:,:,None,None,:]
    #print(f'SAMPLES AFTER {samples.shape}')
    return samples

def matrix_inv(A,eps=0.0,device='cpu', dtype=torch.float32):

  #  print('eps',eps)
    eps = 0.0
    bs,c,h,w,dim,dim = A.shape
    A_reshaped = A.reshape(-1, dim, dim)
    #A_reshaped.dtype=dtype
    # Add a small positive value to the diagonal of each KxK matrix
    A_reshaped += torch.eye(dim, dim).to(device)[None, :, :] * eps
    A_reshaped = torch.tensor(A_reshaped,dtype=dtype)
    # Compute the inverse of each KxK matrix
    inv_A_reshaped = torch.linalg.inv(A_reshaped)

    # Reshape the result back to (bs, 1, 1, 1, K+1, K+1) or bs, 1, 1, 1, 2K, 2K) if paired
    inv_A = inv_A_reshaped.reshape(bs, 1, 1, 1, dim, dim)
    print('dist',torch.dist(A_reshaped[0] @ inv_A_reshaped[0], torch.eye(dim,dtype=dtype)))
    return torch.tensor(inv_A,dtype=torch.float32)

def matrix_vector_mp(A,v):

    '''
    # Example tensors A and v
    A = torch.randn(128, 2, 1, 1, 6, 6)
    v = torch.randn(128, 2, 1, 1, 6)
    '''

    # Reshape v to have an additional dimension at the end (for matrix-vector multiplication)
    v_expanded = v.unsqueeze(-1)  # Now v has shape (128, 2, 1, 1, 6, 1)

    # Perform matrix-vector multiplication
    result = A @ v_expanded

    # The result will have shape (128, 2, 1, 1, 6, 1), you might want to remove the last dimension
    result_squeezed = result.squeeze(-1)  # Now result has shape (128, 2, 1, 1, 6)
    return result_squeezed

def augmented2stacked(z):
    bs, c, h, w, aug_dim = z.shape
    return torch.reshape(z,(bs,int(c*(aug_dim)),h,w))

def stacked2augmented(v, c, aug_dim):
    bs, c_aug, h, w = v.shape
    assert c_aug == int(c*aug_dim)
    return v.reshape(bs,c,h,w,aug_dim)