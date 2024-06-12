def sort_coords(coords):
    mock_filenames = [f"{x}_{y}.jpg" for x, y in coords]
    sorted_filenames = sorted(mock_filenames)
    sorted_coords = [(int(name.split('_')[0]), int(name.split('_')[1].split('.')[0])) for name in sorted_filenames]
    return sorted_coords