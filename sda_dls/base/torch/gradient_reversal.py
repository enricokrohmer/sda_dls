# LICENSE
# This file was extracted from
#   https://github.com/fungtion/DANN
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE

from torch.autograd import Function

class GradientReversalLayer(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
