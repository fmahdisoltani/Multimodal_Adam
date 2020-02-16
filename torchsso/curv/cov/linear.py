import torch
from torchsso import Curvature, DiagCurvature, KronCurvature


class CovLinear(Curvature):

    def update_in_backward(self, grad_output):
        data_input = getattr(self._module, 'data_input', None)  # n x f_in
        assert data_input is not None

        n = data_input.shape[0]

        if self.bias:
            ones = torch.ones((n, 1), device=data_input.device, dtype=data_input.dtype)
            data_input = torch.cat((data_input, ones), 1)  # n x (f_in+1)

        grad = torch.einsum('bi,bj->bij', grad_output, data_input)  # n x f_out x f_in
        grad = grad.reshape((n, -1))  # n x (f_out)(f_in)

        data = torch.einsum('bi,bj->ij', grad, grad)

        self._data = [data]

    def precondition_grad(self, params):
        pass


class DiagCovLinear(DiagCurvature):

    def update_in_backward(self, grad_output):
        data_input = getattr(self._module, 'data_input', None)  # n x f_in
        assert data_input is not None

        n = data_input.shape[0]

        in_in = data_input.mul(data_input)  # n x f_in
        grad_grad = grad_output.mul(grad_output)  # n x f_out

        data_w = torch.einsum('ki,kj->ij', grad_grad,
                              in_in).div(n)  # f_out x f_in
        self._data = [data_w]

        if self.bias:
            data_b = grad_grad.mean(dim=0)  # f_out x 1
            self._data.append(data_b)


class KronCovLinear(KronCurvature):

    def update_in_forward(self, input_data):
        n = input_data.shape[0]  # n x f_in
        if self.bias:
            ones = input_data.new_ones((n, 1))
            # shape: n x (f_in+1)
            input_data = torch.cat((input_data, ones), 1)

        # f_in x f_in or (f_in+1) x (f_in+1)
        A = torch.einsum('ki,kj->ij', input_data, input_data).div(n)
        self._A = A

    def update_in_backward(self, grad_output):
        n = grad_output.shape[0]  # n x f_out

        # f_out x f_out
        G = torch.einsum(
            'ki,kj->ij', grad_output, grad_output).div(n)
        self._G = G

    def precondition_grad(self, params):
        A_inv, G_inv = self.inv

        # todo check params == list?
        if self.bias:
            grad = torch.cat(
                (params[0].grad, params[1].grad.view(-1, 1)), 1)
            preconditioned_grad = G_inv.mm(grad).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad[:, :-1])
            params[1].grad.copy_(preconditioned_grad[:, -1])
        else:
            grad = params[0].grad
            preconditioned_grad = G_inv.mm(grad).mm(A_inv)

            params[0].grad.copy_(preconditioned_grad)

    def sample_params(self, params, mean, std_scale):
        A_ic, G_ic = self.std

        if self.bias:
            m = torch.cat(
                (mean[0], mean[1].view(-1, 1)), 1)
            param = m.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data.copy_(param[:, 0:-1])
            params[1].data.copy_(param[:, -1])
        else:
            m = mean[0]
            param = mean.add(std_scale, G_ic.mm(
                torch.randn_like(m)).mm(A_ic))
            params[0].data = param

    def _get_shape(self):
        linear = self._module
        w = getattr(linear, 'weight')
        f_out, f_in = w.shape

        G_shape = (f_out, f_out)

        if self.bias:
            A_shape = (f_in + 1, f_in + 1)
        else:
            A_shape = (f_in, f_in)

        return A_shape, G_shape


class DiagGMMLinear(DiagCurvature):

    @property
    def shape(self):
        if self._data is None:
            return self._get_shape()

        return tuple([d.shape for d in self._data])

    def _get_shape(self):  # make it gmm yourself
        return tuple(p.shape for p in self.module.parameters())

    def element_wise_init(self, value):  # updated
        self._data = [torch.ones(s, device=self.device).mul(value) for s in self.shape]

    def set_num_gmm(self, num_gmm_components):
        self.num_gmm_components = num_gmm_components

    def sample_params(self, params, mean, std_scale, gmm_pais):
        all_selected_comps = []
        for p, m, std, pai in zip(params, mean, self.std, gmm_pais):  # sample from GMM for each param
            selected_comp = torch.multinomial(torch.tensor(pai), 1)
            noise = torch.randn_like(m[selected_comp])
            p.data.copy_(torch.addcmul(m[selected_comp], std_scale, noise, std[selected_comp]))
            all_selected_comps.append(selected_comp)
        return all_selected_comps

    def sample_params(self, params, mean, std_scale, gmm_pais):
        all_selected_comps = []
        self.num_gmm_components
        for p, m, std, pai in zip(params, mean, self.std, gmm_pais):  # sample from GMM for each param
            # torch.stack([pp.view(-1) for pp in pai])
            stacked_pais = torch.stack(pai).view(self.num_gmm_components, -1)  #torch.stack([pp.view(-1) for pp in pai])
            selected_comp = torch.multinomial(stacked_pais.T, 1)  # 6 numbers
            stacked_means = torch.stack(m).view(self.num_gmm_components, -1)  # ([mm.view(-1) for mm in m])
            stacked_std = torch.stack(std).view(self.num_gmm_components, -1)  # torch.stack([ss.view(-1) for ss in std])
            noise = torch.randn_like(p)
            smean = torch.stack([stacked_means[selected_comp[i], i] for i in range(len(selected_comp))])
            sstd =torch.stack([stacked_std[selected_comp[i], i] for i in range(len(selected_comp))])

            gg = torch.addcmul(smean.reshape_as(noise), std_scale, noise, sstd.reshape_as(noise))
            p.data.copy_(gg)
        return


    def precondition_grad(self, params):
        for p_list, inv_list in zip(params, self.inv):
            for p, inv in zip(p_list, inv_list):  # TODO: using one inv for everyone
                if p.grad is not None:
                    preconditioned_grad = inv.mul(p.grad)
                    p.grad.copy_(preconditioned_grad)

    def update_inv(self): #updated
        ema = self.ema if not self.use_max_ema else self.ema_max
        self.inv = [[self._inv(e) for e in ema_list] for ema_list in ema]

    def update_std(self): #updated
        self.std = [[inv.sqrt() for inv in inv_list] for inv_list in self.inv]

    def update_ema(self): #updated
        data = self.data
        ema = self.ema
        ema_max = self.ema_max
        beta = self.ema_decay
        delta = self.delta

        if ema is None or beta == 1:
            self.ema = [[d.clone() for _ in range(self.num_gmm_components)] for d in self.data]
        else:
            prior_hess = self._l2_reg
            h_hess = self.data  # + prior_hess

            # for e_list,hh in zip(ema, h_hess):
            #     for e in e_list:
            #         (hh * beta).add(e)

            self.ema = [[(hh*beta*d).add(e) for e, d in zip(e_list, d_list)]
                        for e_list, hh, d_list in zip(ema, h_hess, delta._accumulation)]  # update rule
            # self._l2_reg = self._l2_reg * beta * delta #TODO:Farzaneh: Fix

        # if ema is None or beta == 1:
        #     self.ema = [[d.clone() for d in d_list] for d_list in data]
        #     if self.use_max_ema and ema_max is None:
        #         self.ema_max = [[e.clone() for e in e_list]for e_list in self.ema]
        #     self._l2_reg_ema = self._l2_reg
        # else:
        #     self.ema = [[d.mul(beta).add(1 - beta, e) for d, e in zip(d_list, e_list)]
        #                 for d_list, e_list in zip(data, ema)]  # HERE change update rule
        #     self._l2_reg_ema = self._l2_reg * beta + self._l2_reg_ema * (1 - beta)

        if self.use_max_ema:
            for e_list, e_max_list in zip(self.ema, self.ema_max):
                for e, e_max in zip(e_list, e_max_list):
                    torch.max(e, e_max, out=e_max)

    def adjust_data_scale(self, scale):
        self._data = [[d.mul(scale) for d in d_list] for d_list in self._data]

    def update_in_backward(self, grad_output):  # for GMM
        # data_input = getattr(self._module, 'data_input', None)  # n x f_in
        # assert data_input is not None
        #
        # n = data_input.shape[0]
        #
        # in_in = data_input.mul(data_input)  # n x f_in
        # grad_grad = grad_output.mul(grad_output)  # n x f_out
        #
        # data_w_elem = torch.einsum('ki,kj->ij', grad_grad,
        #                       in_in).div(n)  # f_out x f_in
        #
        # data_w = data_w_elem #[data_w_elem for c in range(self.num_gmm_components)]
        #
        # self._data = [data_w]
        #
        #
        # if self.bias:
        #     data_b = grad_grad.mean(dim=0) #[grad_grad.mean(dim=0) for _ in range(self.num_gmm_components)]  # f_out x 1
        #     self._data.append(data_b)

        print("O" * 20)
        print(self.data)
        print(self.data[0].shape)
        print("L" * 20)
        # a = 1 / 0


