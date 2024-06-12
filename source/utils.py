import torch


def sort_coords(coords):
    mock_filenames = [f"{x}_{y}.jpg" for x, y in coords]
    sorted_filenames = sorted(mock_filenames)
    sorted_coords = [(int(name.split('_')[0]), int(name.split('_')[1].split('.')[0])) for name in sorted_filenames]
    return sorted_coords


def track_vram_usage(model, sample):
    # ensure GPU memory is empty before starting
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    # initial memory usage
    initial_memory = torch.cuda.memory_allocated()
    # model inference
    with torch.no_grad():
        output = model(sample)
    # final memory usage
    final_memory = torch.cuda.memory_allocated()
    # memory used for this sample
    memory_used = final_memory - initial_memory
    return output, memory_used