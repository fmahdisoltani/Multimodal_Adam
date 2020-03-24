from torch import Tensor


class TensorAccumulator(object):

    def __init__(self):
        self._accumulation = None

    def check_type(self, data):
        accumulation = self._accumulation

        if isinstance(data, list):
            assert type(data[0]) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'
        else:
            assert type(data) == Tensor, 'the type of data has to be list of torch.Tensor or torch.Tensor'

        if accumulation is not None:
            assert type(data) == type(accumulation), \
                'the type of data ({}) is different from ' \
                'the type of the accumulation ({})'.format(
                    type(data), type(accumulation))

    def update(self, data, scale=1.):
        self.check_type(data)

        accumulation = self._accumulation

        if isinstance(data, list):
            if accumulation is None:
                self._accumulation = [d.mul(scale) for d in data]
            else:
                self._accumulation = [acc.add(scale, d)
                                      for acc, d in zip(accumulation, data)]
        else:
            if accumulation is None:
                self._accumulation = data.mul(scale)
            else:
                self._accumulation = accumulation.add(scale, data)

    def get(self, clear=True):
        accumulation = self._accumulation
        if accumulation is None:
            return

        if isinstance(accumulation, list):
            data = [d.clone() for d in self._accumulation]
        else:
            data = accumulation.clone()

        if clear:
            self.clear()

        return data

    def clear(self):
        self._accumulation = None


class MixtureAccumulator(object):

    def __init__(self, num_gmm_components):
        self._accumulation = None
        self.num_gmm_components = num_gmm_components

    def update(self, data, scale=1.):
        if isinstance(data, list):
            if self._accumulation is None:
                self._accumulation = [[None for _ in range(self.num_gmm_components)] for _ in range(len(data))]

            for d_ind in range(len(data)):
                for comp in range(self.num_gmm_components):
                    if self._accumulation[d_ind][comp] is None:
                        self._accumulation[d_ind][comp] = data[d_ind][comp].clone().mul(scale)
                    else:
                        self._accumulation[d_ind][comp].add_(data[d_ind][comp].mul(scale))

        else:
            if self._accumulation is None:
                self._accumulation = data.mul(scale)
            else:
                self._accumulation = self._accumulation.add(scale, data)

    def get(self, clear=True):
        accumulation = self._accumulation
        if accumulation is None:
            return

        data = [[None for _ in range(self.num_gmm_components)] for _ in accumulation]

        if isinstance(accumulation, list):
            data = [[d.clone() for d in d_list] for d_list in self._accumulation]
        else:
            data = accumulation.clone()

        if clear:
            self.clear()

        return data

    def clear(self):
        self._accumulation = None

