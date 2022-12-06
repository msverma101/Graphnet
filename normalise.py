import torch 
import torch.nn as nn

device = torch.device('cuda')


# class Normalizer(snt.AbstractModule):
class Normalizer(nn.Module):
    def __init__(self, name, range=(0, 1), *, copy=True, clip=False):
        super().__init__()
        self.range = range
        self.copy = copy
        self.clip = clip

    def forward(self, X, accumulate= True, batched= True):
        if accumulate:
            self.partial_fit(X, batched)
        return self.transform(X)

    def fit(self, X, y=None):

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, batched= True):

        range = self.range
        if range[0] >= range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(range)
            )
        
        if batched:
            size= X.shape[2:]
            X= X.reshape((-1, *size))

        first_pass = not hasattr(self, "n_samples_seen_")
        
        data_min, _ = torch.min(X, axis=0)
        data_max, _ = torch.max(X, axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = torch.minimum(self.data_min_, data_min)
            data_max = torch.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (range[1] - range[0]) / self._handle_zeros_in_scale(data_range, copy=True)

        self.min_ = range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):

        X *= self.scale_
        X += self.min_
        if self.clip:
            torch.clip(X, self.range[0], self.range[1], out=X)
        return X
    
    def inverse(self, X):

        X -= self.min_
        X /= self.scale_
        return X

    def _reset(self):

        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def _handle_zeros_in_scale(self, scale, copy=True, constant_mask= None):
       
        if isinstance(scale, torch.Tensor):
            if constant_mask is None:
                # Detect near constant values to avoid dividing by a very small
                # value that could lead to surprising results and numerical
                # stability issues.
                constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

            if copy:
                # New array to avoid side-effects
                scale = scale.clone()
            scale[constant_mask] = 1.0
            return scale
    


