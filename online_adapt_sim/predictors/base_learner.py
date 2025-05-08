import torch
import torch.nn as nn


class LeastSquareProjector(nn.Module):
    def __init__(self, reg, output_dim, device):
        '''
        Base Learner: Weight Least Square Solver

            reg: l2-regulation param
            regress_datalen: data points for regression
            bias_adaptation: whether the adaptive bias is active on the full-connected layer

        Input: the value of encoder basis (vec) and output labels (batch_size, regress_datalen, vec_basis_len + output_size)

        Output: vec matrix (batch_size, vec_matrix_len)
        '''
        super(LeastSquareProjector, self).__init__()
        self.reg = reg
        self.output_dim = output_dim
        self.device = device

    def concat_basis(self, vec_batch):
        # concat the vec basis along diagonal for EACH output labels, and then stack them for batch optimization
        # in: (batch_size, regress_datalen, vec_encoded_dim)
        # out: (batch_size, output_size*regress_len, vec_encoded_dim*output_size)
        batch_list = []
        for vec_regress in vec_batch:
            regress_slices = []
            for vec_ in vec_regress:
                # pick vec_basis only
                slice_ = torch.block_diag(*[vec_[: -self.output_dim]] * self.output_dim)
                regress_slices += [slice_]
            regress_slices_tensor = torch.vstack(regress_slices)
            batch_list += [regress_slices_tensor]

        basis_batch = torch.dstack(batch_list).permute(2, 0, 1)

        return basis_batch

    def concat_label(self, vec_batch):
        # concat the output labels vertically
        # in: (batch_size, regress_datalen, vec_encoded_dim)
        # out: (batch_size, output_size*regress_len, vec_encoded_dim*output_size)
        label_batch = vec_batch[:, :, -self.output_dim :]
        label_batch = label_batch.reshape((label_batch.shape[0], -1, 1))

        return label_batch

    def forward(self, x):
        Phi_N = self.concat_basis(x)
        Label_N = self.concat_label(x)
        batch_eye = (
            torch.dstack([torch.eye(Phi_N.shape[2])] * Phi_N.shape[0])
            .permute(2, 0, 1)
            .to(self.device)
        )
        # closed-form solution of the least square problem
        batch_opt = (
            torch.inverse(Phi_N.mT @ Phi_N + self.reg * batch_eye) @ Phi_N.mT @ Label_N
        )

        return batch_opt
