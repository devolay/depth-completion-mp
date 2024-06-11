import torch

class DepthCompletionMetrics:
    def __init__(self, max_depth=85.0, disp=False, normal=False):
        self.disp = disp
        self.max_depth = max_depth
        self.min_disp = 1.0/max_depth
        self.normal = normal

    def calculate(self, prediction, gt):
        valid_mask = (gt > 0).detach()

        prediction = prediction
        prediction = prediction[valid_mask]
        gt = gt[valid_mask]

        if self.disp:
            prediction = torch.clamp(prediction, min=self.min_disp)
            prediction = 1./prediction
            gt = 1./gt
        if self.normal:
            prediction = prediction * self.max_depth
            gt = gt * self.max_depth
        prediction = torch.clamp(prediction, min=0, max=self.max_depth)

        abs_diff = (prediction - gt).abs()
        rmse = torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
        mae = abs_diff.mean().item()
        return mae, rmse

