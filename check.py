import timm
import inspect
import torch
# Load the Vision Transformer model
model = timm.create_model('vit_base_patch16_224', img_size=32, patch_size=4, num_classes=10, pretrained=True,)

x = torch.randn(1, 3, 32, 32)
output = model(x)
m = torch.nn.Softmax(dim=1)
output = m(output)
print(output.shape)
print(output)
# Get the forward method of the model
#forward_method = model.__init__

# Use inspect to get the source code of the forward method
#forward_source_code = inspect.getsource(forward_method)

#print(forward_source_code)
#print(model)

# print out name of the modules and parameter shapes
"""
# Iterate through all named modules
for module_name, module in model.named_modules():
    print(f"Module: {module_name}")
    
    # Iterate through all named parameters of the module
    for param_name, param in module.named_parameters():
        print(f"\tParameter: {param_name}, Shape: {param.shape}")
"""
"""
# Print out shapes of the parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
"""

"""
all_densenet_models = timm.list_models('*vit_base*')
for model in all_densenet_models:
    print(model)
"""