import timm
import inspect

# Load the Vision Transformer model
model = timm.create_model('vit_base_patch16_224', 
                          img_size=32, 
                          patch_size=4, 
                          num_classes=10, 
                          pretrained=True,
                          )

# Get the forward method of the model
#forward_method = model.__init__

# Use inspect to get the source code of the forward method
#forward_source_code = inspect.getsource(forward_method)

#print(forward_source_code)
#print(model)

# Iterate through all named modules
for module_name, module in model.named_modules():
    print(f"Module: {module_name}")
    
    # Iterate through all named parameters of the module
    for param_name, param in module.named_parameters():
        print(f"\tParameter: {param_name}, Shape: {param.shape}")