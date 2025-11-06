import torch
from model import ResidualDenoiser

if __name__ == "__main__":
    model = ResidualDenoiser()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")

    # Print layer-wise parameter count
    print("\nLayer-wise parameter count:")
    for name, param in model.named_parameters():
        print(f"{name:40s} {param.shape}  params: {param.numel()}")

    # Test input/output dimensions
    x = torch.randn(1, 3, 180, 180)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
