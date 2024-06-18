# import torch
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# else:
#     DEVICE = torch.device('cpu')

# print(f"Device: {DEVICE}")


# import torch
# import torchvision

# device = 'cuda'
# boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
# scores = torch.randn(1).to(device)
# iou_thresholds = 0.5

# print(torchvision.ops.nms(boxes, scores, iou_thresholds))

import torch

def generate_anchor_boxes(feature_map_sizes, scales, aspect_ratios):
    anchor_boxes = []
    for k, f_size in enumerate(feature_map_sizes):
        for i in range(f_size[0]):  # height of the feature map
            for j in range(f_size[1]):  # width of the feature map
                cx = (j + 0.5) / f_size[1]
                cy = (i + 0.5) / f_size[0]
                for ratio in aspect_ratios[k]:
                    for scale in scales[k]:
                        width = scale * torch.sqrt(torch.tensor(ratio))
                        height = scale / torch.sqrt(torch.tensor(ratio))
                        anchor_boxes.append([cx, cy, width, height])
    return torch.tensor(anchor_boxes)

# Example usage:
feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
scales = [[0.1, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1.2]]
aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

anchor_boxes = generate_anchor_boxes(feature_map_sizes, scales, aspect_ratios)
print(anchor_boxes.shape)  # Output the shape of the anchor boxes array
