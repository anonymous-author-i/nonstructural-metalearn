import torch
import torch.nn as nn
from .mlp import *
from .base_learner import LeastSquareProjector


class MLP_Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        hidden_sizes,
        dropout,
        device,
        **kwargs
    ):
        '''
        Encoder : MLP
            How it works:
                1. direct feedforward prediction with current state and control

            [!] This model is currently designed for onestep prediction, i.e., future horizon = 0 is needed
            [!] set history_len = baselearn_len = 0 for dataset preparation (since they are not used)

            Input:
                state-control-label task sequence (current step)
                (batch_size, 1, state_dim + control_dim + label_dim)

            Output:
                predicted output in current time
                (batch_size, 1, output_dim)
        '''

        super(MLP_Encoder, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.mlp_encoder = MLP(
            input_size=state_dim + control_dim,
            hidden_sizes=hidden_sizes,
            output_size=output_dim,
            dropout=dropout,
        )
        self.device = device

    def forward(self, x, args=None):
        x_predict = x[:, -1:, :]
        out = self.mlp_encoder(x_predict[:, :, : -self.output_dim])

        return out


class MLPTimeEmbed_Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        history_len,
        hidden_sizes,
        dropout,
        device,
        **kwargs
    ):
        '''
        Encoder : MLPTimeEmbed
            How it works:
                1. The mlp encoder takes the past + current sequence of state-control pairs to predict the current output

            [!] This model is currently designed for onestep prediction, i.e., future horizon = 0 is needed
            [!] set baselearn_len = 0 for dataset preparation (since they are not used)

            Input:
                state-control sequence (past M step + current step)
                (batch_size, history_len + 1, state_dim + control_dim)

            Output:
                predicted output in current time
                (batch_size, 1, output_dim)

            Settings:
                state_dim: the state dimension of the dynamic system
                control_dim: the control input dimension of the dynamic system
                output_dim: the dimension of prediction output
                history_len: the length of history data input for prediction
                dropout: dropout of neurons in both tcn and mlp
        '''

        super(MLPTimeEmbed_Encoder, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.mlp_encoder = MLP(
            input_size=(state_dim + control_dim) * (history_len + 1),
            hidden_sizes=hidden_sizes,
            output_size=output_dim,
            dropout=dropout,
        )
        self.device = device

    def forward(self, x, args=None):
        x = x[:, :, : self.state_dim + self.control_dim]
        x = x.reshape((x.shape[0], 1, -1))
        out = self.mlp_encoder(x)

        return out


class MetaMLP_Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        base_reg,
        baselearn_len,
        future_horizon,
        hidden_sizes,
        dropout,
        device,
        **kwargs
    ):
        '''
        Meta-Encoder: MLP
            How it works:
                1. The meta-encoder takes current state to predict the current output
                2. The FC layer of the mlp encoder is adaptive and given by a base-learner which takes a past sequence of size M
                to compute the optimal layer parameters, the other layers of the mlp (a basis function) is named as gvec
                3. For current + future prediction, the adaptive layer is fixed.

            [!] set history_len = 0 for dataset preparation (since they are not used)

            Input:
                state-control-label task sequence (base learn N step + current step + future H step)
                (batch_size, sequence_len, state_dim + control_dim + label_dim)

            Output:
                predicted output in current time and future horizon
                (batch_size, future_horizon + 1, output_dim)

            Settings:
                state_dim: the state dimension of the dynamic system
                control_dim: the control input dimension of the dynamic system
                output_dim: the dimension of prediction output
                base_reg: l2-regularizon of base learner
                baselearn_len: data length used in base learner
                future_horizon: the to-be-predicted future horizon (size)
                hidden_sizes: hidden layer sizes in mlp
                kernel_size: conv kernel size in tcn
                dropout: dropout of neurons in both tcn and mlp
        '''

        super(MetaMLP_Encoder, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.baselearn_len = baselearn_len
        self.future_horizon = future_horizon
        self.device = device

        self.mlp_encoder = MLP_withoutFC(
            input_size=state_dim + control_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.base_learner = LeastSquareProjector(base_reg, output_dim, self.device)

    def forward(self, x, args=None):
        '''
        # Notes on dimensions
        # x: (batch_size, baselearn_len + 1 + future_horizon, state_dim + control_dim + label_dim)
        # baselearn_gvec: (batch_size, baselearn_len, mlp_encoded_dim)
        # baselearn_label: (batch_size, baselearn_len, output_dim)
        # coeffvec: (batch_size, output_dim * mlp_encoded_dim)
        # batch_coeff: (batch_size, output_dim, mlp_encoded_dim)
        # gvec_predict: (batch_size, future_horizon + 1, mlp_encoded_dim)
        # coeff_predict: (batch_size, future_horizon + 1, output_dim, mlp_encoded_dim)
        # out: (batch_size, future_horizon + 1, output_dim)
        '''
        x_baselearn = x[:, : x.shape[1] - self.future_horizon - 1, :]

        baselearn_gvec = self.mlp_encoder(x_baselearn[:, :, : -self.output_dim])
        batch_baselearn_label = x_baselearn[:, :, -self.output_dim :]
        base_learner_input = torch.dstack([baselearn_gvec, batch_baselearn_label])

        # get base-learner output
        coeffvec = self.base_learner(base_learner_input)
        batch_coeff = coeffvec.reshape(coeffvec.shape[0], self.output_dim, -1)

        # go prediction in current + future horizon
        x_predict = x[:, -self.future_horizon - 1 :, :]
        gvec_predict = self.mlp_encoder(x_predict[:, :, : -self.output_dim])
        coeff_predict = torch.concat(
            [batch_coeff.unsqueeze(-1)] * (self.future_horizon + 1), axis=3
        ).permute(0, 3, 1, 2)

        # compute output
        out = coeff_predict @ gvec_predict.unsqueeze(-1)

        return out.squeeze(-1), coeff_predict, gvec_predict.unsqueeze(-1)


class MetaMLPTimeEmbed_Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        base_reg,
        history_len,
        baselearn_len,
        future_horizon,
        hidden_sizes,
        dropout,
        device,
        **kwargs
    ):
        '''
        Meta-Encoder: MLPTimeEmbed
            How it works:
                1. The meta-encoder takes past+current state to predict the current output
                2. The FC layer of the mlp encoder is adaptive and given by a base-learner which takes a past sequence of size M
                to compute the optimal layer parameters, the other layers of the mlp (a basis function) is named as gvec
                3. For current + future prediction, the adaptive layer is fixed.

            [!] set history_len = 0 for dataset preparation (since they are not used)

            Input:
                state-control-label task sequence (past M step + base learn N step + current step + future H step)
                (batch_size, sequence_len, state_dim + control_dim + label_dim)

            Output:
                predicted output in current time and future horizon
                (batch_size, future_horizon + 1, output_dim)

            Settings:
                state_dim: the state dimension of the dynamic system
                control_dim: the control input dimension of the dynamic system
                output_dim: the dimension of prediction output
                base_reg: l2-regularizon of base learner
                baselearn_len: data length used in base learner
                future_horizon: the to-be-predicted future horizon (size)
                hidden_sizes: hidden layer sizes in mlp
                kernel_size: conv kernel size in tcn
                dropout: dropout of neurons in both tcn and mlp
        '''

        super(MetaMLPTimeEmbed_Encoder, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.baselearn_len = baselearn_len
        self.future_horizon = future_horizon
        self.device = device

        self.mlp_encoder = MLP_withoutFC(
            input_size=(state_dim + control_dim) * (history_len + 1),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.device = device
        self.base_learner = LeastSquareProjector(base_reg, output_dim, self.device)

    def get_mlp_outputs(self, x_mlpin, rollout_len):
        # x_mlpin: (batch_size, history_len + rollout_len, state_dim + control_dim)
        # return: (batch_size, rollout_len, mlp_encoded_dim)
        mlpin_len = x_mlpin.shape[1] - rollout_len + 1
        gvec_seq = []
        for i in range(rollout_len):
            x_mlp_seq = x_mlpin[:, i : i + mlpin_len, :].reshape(x_mlpin.shape[0], -1)
            gvec = self.mlp_encoder(x_mlp_seq)
            gvec_seq += [gvec]
        return torch.dstack(gvec_seq).permute(0, 2, 1)

    def forward(self, x, args=None):
        '''
        # Notes on dimensions
        # x: (batch_size, baselearn_len + 1 + future_horizon, state_dim + control_dim + label_dim)
        # baselearn_gvec: (batch_size, baselearn_len, mlp_encoded_dim)
        # baselearn_label: (batch_size, baselearn_len, output_dim)
        # coeffvec: (batch_size, output_dim * mlp_encoded_dim)
        # batch_coeff: (batch_size, output_dim, mlp_encoded_dim)
        # gvec_predict: (batch_size, future_horizon + 1, mlp_encoded_dim)
        # coeff_predict: (batch_size, future_horizon + 1, output_dim, mlp_encoded_dim)
        # out: (batch_size, future_horizon + 1, output_dim)
        '''
        x_pastcurrent = x[:, : x.shape[1] - self.future_horizon, :]
        # remove current input for base-learning        
        x_mlpin = x_pastcurrent[:, :-1, : -self.output_dim]
        baselearn_gvec = self.get_mlp_outputs(
            x_mlpin, rollout_len=self.baselearn_len
        )

        # concat hvec with labels for adaptor
        x_baselearn = x_pastcurrent[:, -self.baselearn_len - 1 : -1, :]
        batch_baselearn_label = x_baselearn[:, :, -self.output_dim :]
        base_learner_input = torch.dstack([baselearn_gvec, batch_baselearn_label])

        # get base-learner output
        coeffvec = self.base_learner(base_learner_input)
        batch_coeff = coeffvec.reshape(coeffvec.shape[0], self.output_dim, -1)

        # go prediction in current + future horizon
        x_predict = x[:, self.baselearn_len :, :]
        gvec_predict = self.get_mlp_outputs(x_predict[:, :, :-self.output_dim], rollout_len=self.future_horizon + 1)
        coeff_predict = torch.concat(
            [batch_coeff.unsqueeze(-1)] * (self.future_horizon + 1), axis=3
        ).permute(0, 3, 1, 2)

        # compute output
        out = coeff_predict @ gvec_predict.unsqueeze(-1)

        return out.squeeze(-1), coeff_predict, gvec_predict.unsqueeze(-1)


class MetaMLPtMLP_Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        base_reg,
        history_len,
        baselearn_len,
        future_horizon,
        mlpt_encoder_sizes,
        mlpt_encoded_dim,
        hidden_sizes,
        dropout,
        device,
        **kwargs
    ):
        '''
        Meta-Encoder: MLPt_MLP
            How it works:
                1. The meta-encoder takes the past sequence N of state-control pairs to encode a latent vector (hvec) using mlp_timeEmbed
                2. h_vec is plugged into a mlp encoder together with current state to predict the current output.
                3. The FC layer of the mlp encoder is adaptive and given by a base-learner,
                   which is done by rollout 1 and 2 for past sequence M for base-learner to compute the optimal layer parameters,
                   other mlp and tcn layers before the mlp output is named as gvec.
                4. For current + future prediction, mlp_timeEmbed output and adaptive layer are fixed.

            Input:
                state-control-label task sequence (past M + base learn N step + current step + future H step)
                (batch_size, sequence_len + state_dim + control_dim + label_dim)

            Output:
                predicted output in current time and future horizon
                (batch_size, future_horizon + 1, output_dim)

            Settings:
                state_dim: the state dimension of the dynamic system
                control_dim: the control input dimension of the dynamic system
                output_dim: the dimension of prediction output
                base_reg: l2-regularizon of base learner
                baselearn_len: data length used in base learner
                future_horizon: the to-be-predicted future horizon (size)
                tcn_encoder_sizes: channel sizes in tcn
                tcn_encoded_dim: temporal convolutional output dim
                hidden_sizes: hidden layer sizes in mlp
                dropout: dropout of neurons in both tcn and mlp
        '''

        super(MetaMLPtMLP_Encoder, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.baselearn_len = baselearn_len
        self.future_horizon = future_horizon
        self.device = device

        self.mlpt_encoder = MLPTimeEmbed_Encoder(
            state_dim=state_dim,
            control_dim=control_dim,
            output_dim=mlpt_encoded_dim,
            history_len=history_len,
            hidden_sizes=mlpt_encoder_sizes,
            dropout=dropout,
            device=self.device
        )
        self.mlp_encoder = MLP_withoutFC(
            input_size=state_dim + control_dim + mlpt_encoded_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.base_learner = LeastSquareProjector(base_reg, output_dim, self.device)

    def get_mlpt_outputs(self, x_mlpin, rollout_len):
        # x_mlpin: (batch_size, history_len + rollout_len, state_dim + control_dim)
        # return: (batch_size, rollout_len, mlp_encoded_dim)
        mlpin_len = x_mlpin.shape[1] - rollout_len + 1
        gvec_seq = []
        for i in range(rollout_len):
            x_mlp_seq = x_mlpin[:, i : i + mlpin_len, :]
            gvec = self.mlpt_encoder(x_mlp_seq)
            gvec = torch.squeeze(gvec, dim=1)
            gvec_seq += [gvec]
        return torch.dstack(gvec_seq).permute(0, 2, 1)
    
    def forward(self, x, args=None):
        '''
        # Notes on dimensions
        # x: (batch_size, history_len + baselearn_len + 1 + future_horizon, state_dim + control_dim + label_dim)
        # hvec: (batch_size, tcn_encoded_dim)
        # baselearn_hvec: (batch_size, baselearn_len, tcn_encoded_dim)
        # baselearn_gvec: (batch_size, baselearn_len, mlp_encoded_dim)
        # baselearn_label: (batch_size, baselearn_len, output_dim)
        # coeffvec: (batch_size, output_dim * mlp_encoded_dim)
        # batch_coeff: (batch_size, output_dim, mlp_encoded_dim)
        # mlp_input_predict: (batch_size, future_horizon + 1, state_dim + control_dim + tcn_encoded_dim)
        # gvec_predict: (batch_size, future_horizon + 1, mlp_encoded_dim)
        # coeff_predict: (batch_size, future_horizon + 1, output_dim, mlp_encoded_dim)
        # out: (batch_size, future_horizon + 1, output_dim)
        '''

        # mlpt_encoder takes the past state-control sequence
        x_pastcurrent = x[:, : x.shape[1] - self.future_horizon, :]
        # remove current input for base-learning
        x_mlpin = x_pastcurrent[:, :-1, : -self.output_dim]
        baselearn_hvec = self.get_mlpt_outputs(
            x_mlpin, rollout_len=self.baselearn_len
        )

        # concat hvec with labels for adaptor
        x_baselearn = x_pastcurrent[:, -self.baselearn_len - 1 : -1, :]
        mlp_input_baselearn = torch.concat(
            [x_baselearn[:, :, : -self.output_dim], baselearn_hvec],
            axis=2,
        )
        baselearn_gvec = self.mlp_encoder(mlp_input_baselearn)
        batch_baselearn_label = x_baselearn[:, :, -self.output_dim :]
        base_learner_input = torch.dstack([baselearn_gvec, batch_baselearn_label])

        # get base-learner output
        coeffvec = self.base_learner(base_learner_input)
        batch_coeff = coeffvec.reshape(coeffvec.shape[0], self.output_dim, -1)

        # go prediction in current + future horizon
        x_predict = x[:, -self.future_horizon - 1 :, :]
        hvec_predict = torch.concat(
            [baselearn_hvec[:, -1:, :]] * (self.future_horizon + 1), axis=1
        )
        mlp_input_future = torch.concat(
            [
                x_predict[:, :, : -self.output_dim],
                hvec_predict,
            ],
            axis=2,
        )
        gvec_predict = self.mlp_encoder(mlp_input_future)
        coeff_predict = torch.concat(
            [batch_coeff.unsqueeze(-1)] * (self.future_horizon + 1), axis=3
        ).permute(0, 3, 1, 2)

        # compute output
        out = coeff_predict @ gvec_predict.unsqueeze(-1)

        return out.squeeze(-1)


############################################### Basis Model ######################################
class MetaMLP_Basis(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        hidden_sizes,
        dropout,
        **kwargs
    ):
        super(MetaMLP_Basis, self).__init__()
        self.output_dim = output_dim
        self.mlp_encoder = MLP_withoutFC(
            input_size=state_dim + control_dim,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def forward(self, x, args=None):
        # x: (state_dim + control_dim, 1)
        x = x.reshape(1, -1)
        return self.mlp_encoder(x)


class MetaMLPTimeEmbed_Basis(nn.Module):
    def __init__(
        self,
        state_dim,
        control_dim,
        output_dim,
        history_len,
        hidden_sizes,
        dropout,
        **kwargs
    ):
        super(MetaMLPTimeEmbed_Basis, self).__init__()
        self.output_dim = output_dim
        self.mlp_encoder = MLP_withoutFC(
            input_size=(state_dim + control_dim) * (history_len + 1),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
    
    def forward(self, x, args=None):
        # x: (state_dim + control_dim, history_len + 1)
        x = x.reshape(1, -1)
        return self.mlp_encoder(x)
