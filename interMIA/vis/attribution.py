import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel
from captum.attr import visualization as viz

class Attribution:
    def __init__(self, model):
        self.model = model
        self.model.zero_grad()


    def saliency(self, inp, label):
        attr = Saliency(self.model)
        tensor_attributions = attr.attribute(inp, target=label)
        return tensor_attributions
        pass

    def integrated_gradients(self, inp):
        ig = IntegratedGradients(self.model)
        attr_ig, delta = ig.attribute(inp, baselines=input*0, return_convergence_delta=True)
        return attr_ig, delta

    def noise_tunnel(self, inp):
        attr = NoiseTunnel(self.model)
        tensor_attributions = attr.attribute(inp, baselines=inp*0, nt_type="smoothgrad_sq", nt_samples=100, stdevs=0.2)
        return tensor_attributions

    def deep_lift(self, inp):
        attr = DeepLift(self.model)
        tensor_attributions = attr.attribute(input, baselines=input*0)
        return tensor_attributions


