from torch_points3d.core.data_transform.grid_transform import GridSampling3D

# Custom grid sampling technique for multimodal data, which can be passed in the 'image' multimodal transform section of the wildscenes.yaml file
class GridSampleMultiModal:
    def __init__(self, size=0.05, quantize_coords=True, mode="last"):
        self.grid_sampler = GridSampling3D(size=size, quantize_coords=quantize_coords, mode=mode)

    def __call__(self, data, image_group):
        data = self.grid_sampler(data)
        return data, image_group