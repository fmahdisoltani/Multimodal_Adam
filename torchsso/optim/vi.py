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
            group['mean'] = [[torch.FloatTensor(p.shape).uniform_(-4, 4) for _ in range(num_gmm_components)] for p in group['params']]
            group['pais'] = [[torch.ones_like(p)/num_gmm_components for _ in range(num_gmm_components)]
                             for p in group['params']]

            self.init_buffer(group['mean'])
            group['acc_delta'] = MixtureAccumulator(num_gmm_components)
            group['acc_grads'] = TensorAccumulator()  # [TensorAccumulator()] * num_gmm_components
            group['acc_curv'] = TensorAccumulator()

            if init_precision is not None:
                curv = group['curv']
                curv.set_num_gmm(num_gmm_components)
                curv.element_wise_init(init_precision)
                curv.delta = group['acc_delta']
                curv.step(update_std=(group['std_scale'] > 0))

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

    def calculate_deltas(self, means, stds, pais, params):
        num_gmm_components = len(means[0])
        deltas = []
        for p, mean_list, std_list, pai_list in zip(params, means, stds, pais):
            p_value = p.data.detach()
            down = gmm(p_value, mean_list, std_list, pai_list)
            value = [gaussian(p_value, mean_list[i], std_list[i])/down for i in range(num_gmm_components)]
            if torch.isnan(torch.sum(torch.stack(value))):
                print("inside calculate_deltas")
            deltas.append(value)

        # TODO: check if it is the correct thing to pass std
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
            params, mean = group['params'], group['mean']
            curv = group['curv']
            if curv is not None and curv.std is not None:
                # sample from posterior
                curv.sample_params(params, mean, group['std_scale'], group['pais'])
            else:
                for p, m in zip(params, mean):
                    p.data.copy_(m.data)

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

    def backward_postprocess(self, target='params'):  # acc_grad => group[target].grad
        for group in self.param_groups:
            tt = group[target]

            acc_grads = group['acc_grads'].get()
            for p_list, acc_grad in zip(tt, acc_grads):
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
            for group in self.param_groups:
                params = group['params']
                group['q_entropy'] = [log_gmm(p.data, m_list, s_list, pai_list) for p, m_list, s_list, pai_list
                                      in zip(params, group['mean'], group['curv'].std, group['pais'])]  # pais or log_pais
                ent_loss += torch.sum(torch.stack([torch.sum(g) for g in group['q_entropy']]))

            # forward and backward
            loss, output = closure(ent_loss)
            if torch.isnan(loss):
                print("loss is nan")
            if torch.isnan(torch.sum(output)):
                print("output is nan")
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
                group['acc_grads'].update(grads, scale=1/m/n)

                curv = group['curv']
                if curv is not None:
                    group['acc_curv'].update(curv.data, scale=1/m/n)

                delta = self.calculate_deltas(group['mean'], curv.std, group['pais'], params)
                group['acc_delta'].update(delta)

        loss, prob = acc_loss.get(), acc_prob.get()

        # update acc step
        self.optim_state['acc_step'] += 1
        if self.optim_state['acc_step'] < n:
            return loss, prob
        else:
            self.optim_state['acc_step'] = 0

        self.backward_postprocess(target='mean')
        self.optim_state['step'] += 1

        # update distribution
        for group in self.local_param_groups:

            self.update_preprocess(group, target='mean', grad_type='raw')

            # update covariance
            mean, curv = group['mean'], group['curv']
            if curv is not None:
                curv.update_ema()
                curv.update_std()
                curv.update_inv()
                # curv.step(update_std=(group['std_scale'] > 0))
                # curv.precondition_grad(mean)

            # update mean
            # self.update_preprocess(group, target='mean', grad_type='preconditioned')
            self.update_mean(group)
            # self.update_postprocess(group, target='mean')

            self.update_pais(group, loss)

            # copy mean to param
            params = group['params']
            for p, m_list in zip(params, mean):
                p.data.copy_(m_list[0].data)
                p.grad.copy_(m_list[0].grad)  # TODO: set it to a sample? or the comp with highest pai

        # self.adjust_kl_weighting()
        # print("O"*10)
        # print(group["mean"])
        # print(group["pais"])
        # print(group["curv"].inv)
        # print("P"*10)
        return loss, prob

    def update_mean(self, group):

        means = group['mean']
        params = group['params']
        deltas = group['acc_delta']._accumulation
        invs = group['curv'].inv
        for m_list, d_list, inv_list in zip(means, deltas, invs):
            for m, d, inv in zip(m_list, d_list, inv_list):
                grad = m.grad
                if grad is None:
                    continue
                m.data.add_(-group['lr'], d * grad*inv)  #HERE: * group['ratio']

    def update_pais(self, group, output):
        num_components = self.defaults['num_gmm_components']
        params = group['params']
        deltas = group['acc_delta']._accumulation
        pais = group['pais']
        rhos = [[torch.log(p)-torch.log(p_list[-1]) for p in p_list] for p_list in pais]
        beta = 0.001#self.defaults['lr']
        delta_K = [d_list[-1] for d_list in deltas] #last component for all param

        delta_diff = []
        # for d_list in deltas:
        #     for d, dk in zip(d_list, delta_K):
        #         delta_diff.append(torch.stack([di - dki for di, dki in zip(d, dk)]))
        # rho[c] = (1) * rho[c] - beta * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)

        for d_list, dk in zip(deltas, delta_K):
            delta_diff.append([d - dk for d in d_list])

        rhos = [[(r - d)*output*beta for r, d in zip(r_list, d_list)] for r_list, d_list in zip(rhos, delta_diff)]
        pais = [torch.softmax(torch.stack(r_list), dim=0) for r_list in rhos]
        if torch.isnan(torch.sum(torch.stack(pais))):
            print("booya")

        group['pais'] = [[pai_list[i].data.detach() for i in range(num_components)] for pai_list in pais]

    def update(self, group, target='params'):
        params = group[target]
        for p in params:
            grad = p.grad
            if grad is None:
                continue
            p.data.add_(-group['lr'], grad)

    def update_preprocess(self, group, target='params', grad_type='raw'):
        assert grad_type in ['raw', 'preconditioned'], 'Invalid grad type: {}.'.format(grad_type)
        params = group[target]
        state = self.state

        def apply_gmm_ratio(p, grad):  # delta term
            grad.mul_(group['acc_delta'])

        def apply_gmm_entropy(p, grad):
            grad.add_(group['gmm_entropy'])

        def apply_l2_reg(p, grad):
            if group['l2_reg'] != 0:
                if grad.is_sparse:
                    raise RuntimeError(
                        "l2 regularization option is not compatible with sparse gradients")
                grad.add_(group['l2_reg'], p.data)  # group['l2_reg'] is the regularization rate
                curv = group['curv']
                if curv is not None:
                    curv.l2_reg = group['l2_reg']  # TODO: wtf seriously

        def apply_weight_decay(p, grad):
            if group['weight_decay'] != 0:
                if hasattr(grad, 'is_sparse') and grad.is_sparse:
                    raise RuntimeError(
                        "weight_decay option is not compatible with sparse gradients")
                grad.add_(group['weight_decay'], p.data)

        def apply_momentum(p, grad):
            momentum = group['momentum']

            if momentum != 0:
                buf = state[p]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                grad.copy_(buf)

        def apply_grad_ema_decay(p, grad):
            grad_ema_decay = group['grad_ema_decay']
            if grad_ema_decay != 1:
                buf = state[p]['grad_ema_buffer']
                buf.mul_(1 - grad_ema_decay).add_(grad.mul(grad_ema_decay))
                grad.copy_(buf)

        def apply_bias_correction(grad):
            curv = group['curv']
            beta1 = 1 - group['grad_ema_decay']
            beta2 = 1 - curv.ema_decay

            bias_correction1 = 1 - beta1 ** self.optim_state['step']
            bias_correction2 = 1 - beta2 ** self.optim_state['step']
            if getattr(curv, 'use_sqrt_ema', False):
                bias_correction2 = math.sqrt(bias_correction2)

            grad.mul_(bias_correction2 / bias_correction1)

        def apply_lars(p, grad, thr=1e-2, eps=1e-9):
            d_norm = p.data.norm()
            if d_norm > thr:
                g_norm = grad.norm()
                rate = d_norm / (g_norm + eps)
                grad.mul_(rate)

        for p_list in params:
            for p in p_list:

                grad = p.grad

                if grad is None:
                    continue

                if grad_type == 'raw':
                    apply_l2_reg(p, grad)

                if grad_type == 'preconditioned':
                    apply_weight_decay(p, grad)

                if group['momentum_type'] == grad_type:
                    apply_momentum(p, grad)

                if group['grad_ema_type'] == grad_type:
                    apply_grad_ema_decay(p, grad)

                if grad_type == 'preconditioned' and group['bias_correction']:
                    apply_bias_correction(grad)

                if group['lars_type'] == grad_type and group['lars']:
                    apply_lars(p, grad)

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


def gaussian(x, mean, std):
    return (1 / torch.sqrt(torch.FloatTensor([2*math.pi])*std**2)) * torch.exp(-((x - mean) ** 2.) / (2 * std**2))

def gmm(x, means, stds, pais):
    return sum([pai * gaussian(x, mu, std) for (pai, mu, std) in zip(pais, means, stds)])

def log_gaussian(x, mean, std):
    return -0.5 * torch.log(2 * 3.14 * std ** 2) - (0.5 * (1 / (std ** 2)) * (x - mean) ** 2)

def log_gmm(x, means, stds, log_pais):
    component_log_densities = torch.stack([log_gaussian(x, mu, std) for (mu, std) in zip(means, stds)]).T
    # log_weights = torch.log(pais)
    log_weights = log_normalize(torch.stack(log_pais))
    return torch.logsumexp(component_log_densities + log_weights, axis=-1, keepdims=False)

def log_normalize(x):
    return x - torch.logsumexp(x, 0)