import torch
import numpy as np
import numpy.typing as npt

def run_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample random starting indices
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start, size = batch_size)

    # Create inputs and targets
    inputs = np.stack(dataset[i : i + context_length] for i in starts)
    targets = np.stack(dataset[i + 1 : i + 1 + context_length] for i in starts)

    # Converts to torch tensor
    inputs_tensor = torch.tensor(inputs, dtype= torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor

