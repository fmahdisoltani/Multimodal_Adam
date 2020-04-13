import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsso.optim import SecondOrderOptimizer, DistributedSecondOrderOptimizer
from torchsso.utils import TensorAccumulator, MixtureAccumulator
from torchsso.utils.chainer_communicators import _utility


class VIOptimizer(SecondOrderOptimizer):
    r"""An optimizer for Variational Inference (VI) based on torch.optim.SecondOrderOptimizer.

    This optimizer manages the posterior distribution (mean and covariance of multivariate Gaussian)
        of params for each layer.

    Args:
        model (torch.nn.Module): model with parameters to be trained
        model (float): dataset size
        curv_type (str): type of the curvature ('Hessian', 'Fisher', or 'Cov')
        curv_shapes (dict): shape the curvatures for each type of layer
        curv_kwargs (dict): arguments (with keys) to be passed to torchsso.Curvature.__init__()
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor
        momentum_type (str, optional): type of gradients of which momentum
            is calculated ('raw' or 'preconditioned')
        grad_ema_decay (float, optional): decay rate for EMA of gradients
        grad_ema_type (str, optional): type of gradients of which EMA
            is calculated ('raw' or 'preconditioned')
        weight_decay (float, optional): weight decay
        normalizing_weights (bool, optional): whether the scale of the params
            are normalized after each step
        weight_scale (float, optional): the scale of the params for normalizing weights
        acc_steps (int, optional): number of steps for which gradients and curvatures
            are accumulated before each step
        non_reg_for_bn (bool, optional): whether the regularization is applied to BatchNorm params
        bias_correction (bool, optional): whether the bias correction (refer torch.optim.Adam) is applied
        lars (bool, optional): whether LARS (https://arxiv.org/abs/1708.03888) is applied
        lars_type (str, optional): type of gradients of which LARS
            is applied ('raw' or 'preconditioned')
        num_mc_samples (int, optional): number of MC samples taken from the posterior in each step
        val_num_mc_samples (int, optional): number of MC samples taken from the posterior for evaluation
        kl_weighting (float, optional): KL weighting (https://arxiv.org/abs/1712.02390)
        warmup_kl_weighting_init (float, optional): initial KL weighting for warming up the value
        warmup_kl_weighting_steps (float, optional): number of steps until the value reaches the kl_weighting
        prior_variance (float, optional): variance of the prior distribution (Gaussian) of each param
        init_precision (float, optional): initial (diagonal) precision of the posterior of params
    """

    def __init__(self, model: nn.Module, dataset_size: float, curv_type: str, curv_shapes: dict, curv_kwargs: dict,
                 num_gmm_components=1,
                 lr=0.01, momentum=0., momentum_type='preconditioned',
                 grad_ema_decay=1., grad_ema_type='raw', weight_decay=0.,
                 normalizing_weights=False, weight_scale=None,
                 acc_steps=1, non_reg_for_bn=False, bias_correction=False,
                 lars=False, lars_type='preconditioned',
                 num_mc_samples=10, val_num_mc_samples=10,
                 kl_weighting=1, warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=None,
                 prior_variance=1, init_precision=None,
                 seed=1, total_steps=1000):

        if dataset_size < 0:
            raise ValueError("Invalid dataset size: {}".format(dataset_size))
        if num_mc_samples < 1:
            raise ValueError("Invalid number of MC samples: {}".format(num_mc_samples))
        if val_num_mc_samples < 0:
            raise ValueError("Invalid number of MC samples for validation: {}".format(val_num_mc_samples))
        if kl_weighting < 0:
            raise ValueError("Invalid KL weighting: {}".format(kl_weighting))
        if warmup_kl_weighting_steps is not None and warmup_kl_weighting_init < 0:
            raise ValueError("Invalid initial KL weighting: {}".format(warmup_kl_weighting_init))
        if prior_variance < 0:
            raise ValueError("Invalid prior variance: {}".format(prior_variance))
        if init_precision is not None and init_precision < 0:
            raise ValueError("Invalid initial precision: {}".format(init_precision))

        init_kl_weighting = kl_weighting if warmup_kl_weighting_steps is None else warmup_kl_weighting_init
        l2_reg = init_kl_weighting / dataset_size / prior_variance if prior_variance != 0 else 0
        std_scale = math.sqrt(init_kl_weighting / dataset_size)

        super(VIOptimizer, self).__init__(model, curv_type, curv_shapes, curv_kwargs,
                                          lr=lr, momentum=momentum, momentum_type=momentum_type,
                                          grad_ema_decay=grad_ema_decay, grad_ema_type=grad_ema_type,
                                          l2_reg=l2_reg, weight_decay=weight_decay,
                                          normalizing_weights=normalizing_weights, weight_scale=weight_scale,
                                          acc_steps=acc_steps, non_reg_for_bn=non_reg_for_bn,
                                          bias_correction=bias_correction,
                                          lars=lars, lars_type=lars_type)

        self.num_gmm_components = num_gmm_components
        self.defaults['std_scale'] = std_scale
        self.defaults['num_gmm_components'] = num_gmm_components
        self.defaults['kl_weighting'] = kl_weighting
        self.defaults['warmup_kl_weighting_init'] = warmup_kl_weighting_init
        self.defaults['warmup_kl_weighting_steps'] = warmup_kl_weighting_steps
        self.defaults['num_mc_samples'] = num_mc_samples
        self.defaults['val_num_mc_samples'] = val_num_mc_samples
        self.defaults['total_steps'] = total_steps
        self.defaults['seed_base'] = seed

        for group in self.param_groups:
            group['std_scale'] = 0 if group['l2_reg'] == 0 else std_scale
            # group['mean'] = [[torch.ones_like(p)*0.3 for _ in range(num_gmm_components)] for p in group['params']]
            group['mean'] = [[p.data.detach().clone()+i*.1 for i in range(num_gmm_components)] for p in group['params']]

            # group['mean'] = [[torch.FloatTensor(p.shape).uniform_(-4, 4) for _ in range(num_gmm_components)] for p in group['params']]
            group['prec'] = [[torch.ones_like(p) * init_precision for _ in
                              range(num_gmm_components)] for p in
                             group['params']]
            self.update_cov(group)
            group['pais'] = [[torch.ones_like(p)/num_gmm_components for _ in range(num_gmm_components)]
                             for p in group['params']]

            self.init_buffer(group['mean'])
            group['acc_delta'] = MixtureAccumulator(num_gmm_components)
            group['acc_grads'] = TensorAccumulator()  # [TensorAccumulator()] * num_gmm_components
            group['acc_curv'] = TensorAccumulator()

            if init_precision is not None:
                curv = group['curv']
                curv.element_wise_init(init_precision)

    def init_buffer(self, params):
        for p_list in params:
            # if isinstance(p, list):
            for p in p_list:
                state = self.state[p] # TODO: Question
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['grad_ema_buffer'] = torch.zeros_like(p.data)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tenfsor` s."""
        for group in self.param_groups:
            for m_list in group['mean']:
                for m in m_list:
                    if m.grad is not None:
                        m.grad.detach_()
                        m.grad.zero_()

        super(VIOptimizer, self).zero_grad()

    def calculate_deltas(self, means, covs, pais, params):
        num_gmm_components = len(means[0])
        deltas = []
        for p, mean_list, cov_list, pai_list in zip(params, means, covs, pais):
            p_value = p.data.detach()
            down = gmm(p_value, mean_list, cov_list, pai_list)
            value = [gaussian(p_value, mean_list[i], cov_list[i])/down for i in range(num_gmm_components)]
            if torch.isnan(torch.sum(torch.stack(value))):
                print("inside calculate_deltas")
            deltas.append(value)

        return deltas

    @property
    def seed(self):
        return self.optim_state['step'] + self.defaults['seed_base']

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_params(self):

        for group in self.param_groups:
            std_scale = group['std_scale']

            for params, means, covs, pais in zip(group['params'], group['mean'],
                                                 group['cov'], group['pais']):  # sample from GMM for each param
                # torch.stack([pp.view(-1) for pp in pai])
                # noise = torch.randn_like(means[0])
                # params.data.copy_(
                #     torch.addcmul(means[0], group['std_scale'], noise,
                #                   torch.sqrt(covs[0])))
                noise = torch.randn_like(params)
                stacked_pais = torch.stack(pais).view(self.num_gmm_components, -1)
                selected_comp = torch.multinomial(stacked_pais.T, 1)
                # selected_comp = torch.zeros_like(selected_comp)
                stacked_means = torch.stack(means).view(self.num_gmm_components, -1)  # ([mm.view(-1) for mm in m])
                stacked_cv = torch.stack(covs).view(self.num_gmm_components, -1)  # torch.stack([ss.view(-1) for ss in std])
                mask = torch.zeros_like(stacked_means).scatter_(0,selected_comp.T, 1.)
                selected_mean = torch.sum(stacked_means.mul(mask), dim=0)

                selected_cov = torch.sum(stacked_cv.mul(mask), dim=0)
                std = torch.sqrt(selected_cov)

                gg = torch.addcmul(selected_mean.reshape_as(noise), std_scale, noise,
                                   std.reshape_as(noise))
                params.data.copy_(gg)
                # print("%%%%% Sample %%%%%")
                # print(gg)
                # p.data.copy_(m[0])
    def sample_params1(self):

        for group in self.param_groups:
            # sample from posterior
            for params, means, covs in zip(group['params'], group['mean'], group['cov']):

                noise = torch.randn_like(params)
                params.data.copy_(torch.addcmul(means[0],  group['std_scale'], noise, torch.sqrt(covs[0])))


    def copy_mean_to_params(self):
        for group in self.param_groups:
            params, mean = group['params'], group['mean']
            for p, m in zip(params, mean):
                p.data.copy_(m[0].data)
                if getattr(p, 'grad', None) is not None \
                        and getattr(m, 'grad', None) is not None:
                    p.grad.copy_(m.grad)

    def adjust_kl_weighting(self):
        warmup_steps = self.defaults['warmup_kl_weighting_steps']
        if warmup_steps is None:
            return

        current_step = self.optim_state['step']
        if warmup_steps < current_step:
            return

        target_kl = self.defaults['kl_weighting']
        init_kl = self.defaults['warmup_kl_weighting_init']

        rate = current_step / warmup_steps
        kl_weighting = init_kl + rate * (target_kl - init_kl)

        rate = kl_weighting / init_kl
        l2_reg = rate * self.defaults['l2_reg']
        std_scale = math.sqrt(rate) * self.defaults['std_scale']
        for group in self.param_groups:
            if group['l2_reg'] > 0:
                group['l2_reg'] = l2_reg
            if group['std_scale'] > 0:
                group['std_scale'] = std_scale

    def backward_postprocess(self):  # acc_grad => group[target].grad
        for group in self.param_groups:
            acc_grads = group['acc_grads'].get()
            for p_list, acc_grad in zip(group['mean'], acc_grads):
                for p in p_list:
                    if acc_grad is not None:
                        p.grad = acc_grad.clone()

            curv = group['curv']
            if curv is not None:
                curv.data = group['acc_curv'].get()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        def closure():
            # forward/backward
            return loss, output
        """

        m = self.defaults['num_mc_samples']
        n = self.defaults['acc_steps']

        acc_loss = TensorAccumulator()
        acc_prob = TensorAccumulator()

        self.set_random_seed()

        for _ in range(m):

            # sampling
            self.sample_params()

            # forward and backward
            ent_loss = 0
            reg_loss = 0
            for group in self.param_groups:
                params = group['params']
                group['q_entropy'] = [log_gmm(p, m_list, s_list, pai_list) for p, m_list, s_list, pai_list
                                      in zip(params, group['mean'], group['cov'], group['pais'])]  # pais or log_pais
                ent_loss += torch.sum(torch.stack([torch.sum(g) for g in group['q_entropy']]))
                reg_loss += sum([torch.sum(group['l2_reg'] * p.data ** 2) for p in params])
                # reg_loss += torch.sum(torch.stack([group['l2_reg'] * p.data ** 2 for p in params]))

            surrogate_loss = ent_loss-reg_loss
            loss, output, network_loss = closure(surrogate_loss)
            # for p in params:
            #     print(p.grad)
                # p.grad.add_(group['l2_reg'], p.data)  # Add derivative of prior

            acc_loss.update(loss, scale=1/m)
            if output.ndim == 2:
                prob = F.softmax(output, dim=1)
            elif output.ndim == 1:
                prob = torch.sigmoid(output)
            else:
                raise ValueError(f'Invalid ndim {output.ndim}')
            acc_prob.update(prob, scale=1/n)

            # accumulate
            for group in self.param_groups:
                params = group['params']
                grads = [p.grad.data for p in params]
                # print("%%%%%%%%%%%% this is grad %%%%%%%%%%%")
                # print(grads)
                group['acc_grads'].update(grads, scale=1/m/n)
                group['acc_curv'].update(group['curv'].data, scale=1/m/n)
                delta = self.calculate_deltas(group['mean'],
                                              group['cov'], group['pais'], params)
                group['acc_delta'].update(delta, scale=1/m/n)

        loss, prob = acc_loss.get(), acc_prob.get()

        # update acc step
        self.optim_state['acc_step'] += 1
        if self.optim_state['acc_step'] < n:
            return loss, prob
        else:
            self.optim_state['acc_step'] = 0

        self.backward_postprocess()
        self.optim_state['step'] += 1

        # update distribution
        for group in self.local_param_groups:
            # print("O" * 10)
            # print(group["mean"])
            # print(group["pais"])
            # print(group["cov"])
            # print("P" * 10)
            deltas = group['acc_delta'].get()
            self.update_prec(group, deltas)
            self.update_cov(group)
            self.update_mean(group, deltas)
            self.update_pais(group, loss, deltas)

            # copy mean to param
            params = group['params']
            for p, m_list in zip(params,  group['mean']):
                p.data.copy_(m_list[0].data)
                p.grad.copy_(m_list[0].grad)  # TODO: set it to a sample? or the comp with highest pai

            self.adjust_kl_weighting()


        return loss, prob, network_loss

    def update_prec(self, group, deltas):
        # prec = group['prec']
        beta = 0.01
        # delta = group['acc_delta']

        if group['prec'] is None or beta == 1:
            group['prec'] = [[d.clone() for _ in range(self.num_gmm_components)] for d in group['curv'].data]
        else:
            h_hess = group['curv'].data

            group['prec'] = [[(hh*beta*d).add(e) for e, d in zip(e_list, d_list)]
                        for e_list, hh, d_list in zip(group['prec'] , h_hess, deltas)]  # update rule

    def update_cov(self, group):
        group['cov'] = [[1 / e for e in prec_list] for prec_list in group['prec']]

    def update_mean(self, group, deltas):
        means = group['mean']
        # deltas = group['acc_delta']._accumulation
        cov = group['cov']
        for m_list, d_list, cov_list in zip(means, deltas, cov):
            for m, d, inv in zip(m_list, d_list, cov_list):
                grad = m.grad
                m.data.add_(-group['lr'], d * grad * inv)  #HERE: * group['ratio']
                # print("%%%%%%%% update value of mean %%%%%%%%")
                # print(d * grad * inv)
                # print(inv)
                # print(grad)
                # print(d)

    def update_pais(self, group, output, deltas):
        num_components = self.defaults['num_gmm_components']
        # deltas = group['acc_delta']._accumulation
        pais1 = group['pais']
        rhos1 = [[torch.log(p)-torch.log(p_list[-1]) for p in p_list] for p_list in pais1]
        # beta = 0.001#self.defaults['lr']
        delta_K = [d_list[-1] for d_list in deltas] #last component for all param

        delta_diff = []

        for d_list, dk in zip(deltas, delta_K):
            delta_diff.append([d - dk for d in d_list])

        rhos = [[(r - d)*output*group['lr'] for r, d in zip(r_list, d_list)] for r_list, d_list in zip(rhos1, delta_diff)]
        pais = [torch.softmax(torch.stack(r_list), dim=0) for r_list in rhos]
        # if torch.isnan(torch.sum(torch.stack(pais))):
        if torch.isnan(sum([torch.sum(p) for p in pais])):
            print("booya")

        group['pais'] = [[pai_list[i].data.detach() for i in range(num_components)] for pai_list in pais]

    def prediction(self, data, mc=None, keep_probs=False):

        self.set_random_seed(self.optim_state['step'])

        acc_prob = TensorAccumulator()
        probs = []

        mc_samples = self.defaults['val_num_mc_samples'] if mc is None else mc

        use_mean = mc_samples == 0
        n = 1 if use_mean else mc_samples

        for _ in range(n):

            if use_mean:
                self.copy_mean_to_params()
            else:
                # sampling
                self.sample_params()

            output = self.model(data)
            if output.ndim == 2:
                prob = F.softmax(output, dim=1)
            elif output.ndim == 1:
                prob = torch.sigmoid(output)
            else:
                raise ValueError(f'Invalid ndim {output.ndim}')

            acc_prob.update(prob, scale=1/n)
            if keep_probs:
                probs.append(prob)

        self.copy_mean_to_params()

        prob = acc_prob.get()

        if keep_probs:
            return prob, probs
        else:
            return prob


class VOGN(VIOptimizer):

    def __init__(self, *args, **kwargs):
        default_kwargs = dict(lr=1e-3,
                              curv_type='Cov',
                              curv_shapes={
                                  'Linear': 'Diag',
                                  'Conv2d': 'Diag',
                                  'BatchNorm1d': 'Diag',
                                  'BatchNorm2d': 'Diag'
                              },
                              curv_kwargs={'ema_decay': 0.01, 'damping': 1e-7},
                              warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=1000,
                              grad_ema_decay=0.1, num_mc_samples=50, val_num_mc_samples=100)

        default_kwargs.update(kwargs)

        super(VOGN, self).__init__(*args, **default_kwargs)


class DistributedVIOptimizer(DistributedSecondOrderOptimizer, VIOptimizer):

    def __init__(self, *args, mc_group_id=0, **kwargs):
        super(DistributedVIOptimizer, self).__init__(*args, **kwargs)
        self.defaults['seed_base'] += mc_group_id * self.defaults['total_steps']

    @property
    def actual_optimizer(self):
        return VIOptimizer

    def zero_grad(self):
        self.actual_optimizer.zero_grad(self)

    def extractors_for_rsv(self):
        extractors = [_utility.extract_attr_from_params('grad', target='mean'),
                      _utility.extract_attr_from_curv('data', True)]
        return extractors

    def extractors_for_agv(self):
        extractors = [_utility.extract_attr_from_params('data', target='mean'),
                      _utility.extract_attr_from_curv('std', True)]
        return extractors

    def step(self, closure=None):
        ret = super(DistributedVIOptimizer, self).step(closure)

        if self.is_updated():
            self.copy_mean_to_params()

        return ret


def gaussian(x, mean, cov):
    return (1 / torch.sqrt(torch.FloatTensor([2*math.pi])*cov)) * torch.exp(-((x - mean) ** 2.) / (2 * cov))

def gmm(x, means, covs, pais):
    return sum([pai * gaussian(x, mu, cov) for (pai, mu, cov) in zip(pais, means, covs)])

def log_gaussian(x, mean, cov):
    return -0.5 * torch.log(2 * 3.14 * cov) - (0.5 * (1 / (cov)) * (x - mean) ** 2)

def log_gmm(x, means, covs, log_pais):
    component_log_densities = torch.stack([log_gaussian(x, mu, cov) for (mu, cov) in zip(means, covs)])
    # component_log_densities = torch.transpose(component_log_densities, dim0=1, dim1=0)
    # log_weights = torch.log(pais)
    log_weights = log_normalize(torch.stack(log_pais))
    return torch.logsumexp(component_log_densities + log_weights, axis=-1, keepdims=False)

def log_normalize(x):
    return x - torch.logsumexp(x, 0)