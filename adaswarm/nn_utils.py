import torch
import torch.nn.functional as F
from adaswarm.rempso import RotatedEMParticleSwarmOptimizer


class CELossWithPSO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_pred, swarm_learning_rate=0.1):
        particle_swarm_optimizer = RotatedEMParticleSwarmOptimizer(
            dimension=125, swarm_size=10, number_of_classes=10, targets=y_pred
        )
        sum_cr, gbest = particle_swarm_optimizer.run_iteration(number=5)
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = swarm_learning_rate
        ctx.gbest = gbest
        return F.cross_entropy(y, y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr / eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None


class L1Loss:
    def __init__(self, y):
        self.y = y
        self.fitness = torch.nn.L1Loss()

    def evaluate(self, x):
        # print(x, self.y)
        return self.fitness(x, self.y)


class L1LossWithPSO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, y_pred, sum_cr, eta, gbest):
        ctx.save_for_backward(y, y_pred)
        ctx.sum_cr = sum_cr
        ctx.eta = eta
        ctx.gbest = gbest
        return F.l1_loss(y, y_pred)

    @staticmethod
    def backward(ctx, grad_output):
        yy, yy_pred = ctx.saved_tensors
        sum_cr = ctx.sum_cr
        eta = ctx.eta
        grad_input = torch.neg((sum_cr / eta) * (ctx.gbest - yy))
        return grad_input, grad_output, None, None, None