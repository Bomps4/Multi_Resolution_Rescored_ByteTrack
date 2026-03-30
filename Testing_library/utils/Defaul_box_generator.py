from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from typing import List,Union,Tuple,Callable,Dict,Optional
import torch



class MyDefaultBoxGenerator (DefaultBoxGenerator):
    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__(aspect_ratios,min_ratio,max_ratio,scales,steps,clip)

    
    def forward(self,inp_tensor,feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = inp_tensor.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        x_y_size = torch.tensor([1,1], device=default_boxes.device)
        for _ in inp_tensor:
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                    (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                ],
                -1,
            )
            dboxes.append(dboxes_in_image)
        return dboxes
