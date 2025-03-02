import torch
import math
import logging
import time
import utils


class GPTQ_bf16:

    def __init__(
        self, 
        layer, 
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.batch_index = 0 # dummy

    def add_batch(self, inp, out, weighting=None):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        if weighting is not None:
            # normalize weighting
            weighting = weighting / weighting.sum() * weighting.shape[0]
            inp = inp * weighting.to(inp.device) ** 0.5
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())

        self.H += inp.matmul(inp.t())
        
    def _channelwise_squared_error(self, XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
        """
        Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
        :note: if XTX is divided by dataset size, this function will return *mean* squared error
        :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
        :param reference_weight: reference weight of shape [out_features, in_features]
        :return: per-channel squared errors of shape [*dims, out_features]
        """
        XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
        XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
        dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
        return XW_norm_square - 2 * dot_product + XWreference_norm_square

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, quant=True,
        original_dtype=torch.bfloat16,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
    
        multiplier = 1
        
        original_H = H.clone()
        original_W = W.clone()
        
        original_loss = self._channelwise_squared_error(
            original_H.to(self.dev), 
            original_W.to(original_dtype).float(),
            original_W.to(self.dev),
        )
        
        while multiplier < 50:
            try:
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                break
            except:
                multiplier += 1
                
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.to(original_dtype)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
            
        new_loss = self._channelwise_squared_error(
            original_H.to(self.dev), 
            Q,
            original_W.to(self.dev),
        )
        
        print(original_loss.sum(), new_loss.sum())
            
        self.layer.weight.data = Q.reshape(self.layer.weight.shape)
        # if hasattr(self.layer, 'bias') and self.layer.bias is not None:
        #     self.layer.bias.data = self.layer.bias.data.to(original_dtype)

        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)