import torch
import pprint

class Affine(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        return self.linear(input) + self.bias

affine = Affine(3, 4)

pp = pprint.PrettyPrinter(width=41, compact=True)
named_str = ["affine.named_buffers", "affine.named_children", "affine.named_modules", "affine.named_parameters"]
named = [affine.named_buffers, affine.named_children, affine.named_modules, affine.named_parameters]
for name_str, name in zip(named_str, named):
    print(name_str)
    pp.pprint(list(name()))
    print()

print("in")
in_tensor = torch.tensor([10000.,100.,1.])
print(in_tensor)
print("out")
print(affine(in_tensor))


