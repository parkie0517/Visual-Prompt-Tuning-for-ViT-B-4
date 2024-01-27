import timm
import inspect

# Load the pre-trained model
pretrained_model_name = 'vit_base_patch16_224_in21k'
model = timm.create_model(pretrained_model_name, pretrained=True)
"""
# Iterate through all named modules
for module_name, module in model.named_modules():
    print(f"Module: {module_name}")
    
    # Iterate through all named parameters of the module
    for param_name, param in module.named_parameters():
        print(f"\tParameter: {param_name}, Shape: {param.shape}")
"""
print(inspect.getsource(model.forward.forward_features))