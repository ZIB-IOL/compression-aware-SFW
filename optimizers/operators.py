# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         operators.py
# Description:  Proximal Operator
# ===========================================================================
import torch

class ProximalOperator:
    """Static class containing proximal operators, each function returns a function, i.e. the proximal operator."""
    @staticmethod
    def svd_soft_thresholding(threshold):
        """Implements Soft-Thresholding on the singular values"""
        @torch.no_grad()
        def operator(x, lr):
            with torch.no_grad():
                W = x
                if len(W.shape) == 4:
                    W = W.flatten(start_dim=1)

                # This often fails with the following bug, hence we try several methods until we get convergence
                # "RuntimeError: svd_cuda: (Batch element 0): The algorithm failed to converge because the input
                # matrix is ill-conditioned or has too many repeated singular values (error code: 129). "
                # Note: This seems to be a CUDA only issue, moving to CPU solves the problem
                n_methods = 8 if torch.cuda.is_available() else 4
                dev = W.device
                for method in range(n_methods):
                    selection_method = method
                    if method//4 == 1:
                        # Move to cpu
                        W = W.to(device='cpu')
                        selection_method = method % 4
                    try:
                        if selection_method == 0:
                            U, S, V_t = torch.linalg.svd(W, full_matrices=False)
                        elif selection_method == 1:
                            U, S, V_t = torch.svd(W, some=True)
                        elif selection_method == 2:
                            # Perturb the matrix as suggested in https://github.com/pytorch/pytorch/issues/28293
                            U, S, V_t = torch.linalg.svd(W + 1e-4 * W.mean() * torch.rand_like(W), full_matrices=False)
                        elif selection_method == 3:
                            # Perturb the matrix as suggested in https://github.com/pytorch/pytorch/issues/28293
                            U, S, V_t = torch.svd(W + 1e-4 * W.mean() * torch.rand_like(W), some=True)

                        break
                    except Exception as e:
                        print(e)
                        continue

                # Potentially move everything back to GPU
                W = W.to(device=dev)
                U = U.to(device=dev)
                S = S.to(device=dev)
                V_t = V_t.to(device=dev)

                # Threshold S
                S = torch.nn.functional.relu(S-lr*threshold)

            # Recompute matrix from truncated SVD
            return torch.mm(U, torch.mm(torch.diag(S), V_t)).view(x.shape)

        return operator
