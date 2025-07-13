import numpy as np
import PIL
from PIL import Image
import torch


def pil_to_tensor(
    img: Image.Image,
    target_image_size=512,
    lock_ratio=True,
    center_crop=True,
    padding=False,
    standardize=True,
    **kwarg
) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")

    if isinstance(target_image_size, int):
        target_size = (target_image_size, target_image_size)
        if target_image_size < 0:
            target_size = img.size
    else:
        target_size = target_image_size  # (width, height)

    if lock_ratio:
        original_width, original_height = img.size
        target_width, target_height = target_size

        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if center_crop:
            scale = max(scale_w, scale_h)
        elif padding:
            scale = min(scale_w, scale_h)
        else:
            scale = 1.0  # fallback

        new_size = (round(original_width * scale), round(original_height * scale))
        img = img.resize(new_size, Image.LANCZOS)

        if center_crop:
            left = (img.width - target_width) // 2
            top = (img.height - target_height) // 2
            img = img.crop((left, top, left + target_width, top + target_height))
        elif padding:
            new_img = Image.new("RGB", target_size, (0, 0, 0))
            left = (target_width - img.width) // 2
            top = (target_height - img.height) // 2
            new_img.paste(img, (left, top))
            img = new_img
    else:
        img = img.resize(target_size, Image.LANCZOS)

    np_img = np.array(img) / 255.0  # Normalize to [0, 1]
    if standardize:
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()  # (C, H, W)

    return tensor_img


def tensor_to_pil(chw_tensor: torch.Tensor, standardize=True, **kwarg) -> PIL.Image:
    # Ensure tensor is on CPU and detached (avoid if already satisfied)
    if chw_tensor.device.type != "cpu" or chw_tensor.requires_grad:
        chw_tensor = chw_tensor.detach().cpu()

    # Normalize tensor to [0,1]
    if standardize:
        chw_tensor = torch.clamp(chw_tensor, -1.0, 1.0).add_(1).div_(2)
    else:
        chw_tensor = torch.clamp(chw_tensor, 0.0, 1.0)

    # Convert to HWC; make contiguous for fast numpy conversion
    hwc = chw_tensor.permute(1, 2, 0).contiguous()

    # Convert to numpy and uint8 in one step
    array = hwc.mul(255).to(torch.uint8).numpy()

    # Convert NumPy array directly to PIL Image.
    pil_image = Image.fromarray(array)

    # Convert image to RGB only if needed
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    return pil_image
