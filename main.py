import torch
from torch import Tensor
from torch.utils.data import DataLoader
from mmfall.data import MMFallDataset
from mmfall.model import MMFall


def transform(tensor: Tensor) -> Tensor:
    dim = tensor.size(-1)
    assert tensor.size(0) <= 64
    pad_size = 64 - tensor.size(0)
    return torch.cat((tensor, torch.zeros(pad_size, dim)), dim=0)[:, :4]


def main(_: list[str]) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    # Load the dataset.
    dataset      = MMFallDataset("dataset/people_tracking/walking.bin", pattern_size=26, device=device, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader  = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create the model.
    model = MMFall(num_frames=26, num_points=64, channels=4)
    print(model)

    model.fit(train_loader, test_loader, epochs=200)


if __name__ == "__main__":
    import sys
    main(sys.argv)
