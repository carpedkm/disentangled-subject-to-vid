import os
import json
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

from torch.utils.data import Dataset

# ✅ Load and filter the dataset
print("Loading and filtering the dataset...")
dataset = load_dataset("/root/daneul/projects/refactored/CogVideo/Subjects200K_collection3")

def filter_func(item):
    if not item.get("quality_assessment"):
        return False
    return all(
        item["quality_assessment"].get(key, 0) >= 5
        for key in ["compositeStructure", "objectConsistency", "imageQuality"]
    )

data_valid = dataset["train"].filter(
    filter_func, num_proc=16, cache_file_name="./cache/dataset/data_valid_1024.arrow"
)

# ✅ Initialize the dataset class
class Subject200KDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        item = self.base_dataset[idx // 2]
        image = item["image"]

        # Crop and resize images
        padding = 0
        left_img = image.crop((padding, padding, 1024 + padding, 1024 + padding)).resize((720, 720)).convert("RGB")
        # right_img = image.crop((1024 + 2 * padding, padding, 1024 + 2 * padding, 1024 + padding)).resize((720, 720)).convert("RGB")
        right_img = image.crop((1024 + padding, padding, 2048 + padding, 1024 + padding)).resize((720, 720)).convert("RGB")
        width, height = 720, 720
        target_width, target_height = 720, 480

        # Calculate coordinates for center crop
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        # Center-crop the image
        left_img = left_img.crop((left, top, right, bottom))
        right_img = right_img.crop((left, top, right, bottom))
        description_0 = item["description"]["description_0"]
        description_1 = item["description"]["description_1"]
        # Safely access nested fields, replace None with 0
        item_desc = item.get("description", {})
        return {
            "left_image": self.to_tensor(left_img),
            "right_image": self.to_tensor(right_img),
            "description_0": description_0,
            "description_1": description_1,
            "item": item_desc.get("item", 0),
            "category": item_desc.get("category", 0),
            "collection": item.get("collection", 0),
            "quality_assessment": item["quality_assessment"],
        }

subject_dataset = Subject200KDataset(data_valid)

import torch
def collate_fn(batch):
    # category none check
    for item in batch:
        if item["category"] is None:
            item["category"] = 0
        # item none check
        if item["item"] is None:
            item["item"] = 0
        # collection none check
        if item["collection"] is None:
            item["collection"] = 0
        # NONE detected
        # print('None detected in category, item, collection')
    return {
        "left_image": torch.stack([item["left_image"] for item in batch]),
        "right_image": torch.stack([item["right_image"] for item in batch]),
        "description_0": [item["description_0"] for item in batch],
        "description_1": [item["description_1"] for item in batch],
        "item": [item["item"] for item in batch],
        "category": [item["category"] for item in batch],
        "collection": [item["collection"] for item in batch
        ],
        "quality_assessment": {
            key: torch.tensor([item["quality_assessment"][key] for item in batch]) 
            for key in batch[0]["quality_assessment"]
        },
    }

# ✅ Use DataLoader for batch processing
dataloader = DataLoader(subject_dataset, batch_size=480, num_workers=64, collate_fn=collate_fn)

# ✅ Create output directories
os.makedirs("output_720_1024/left_images_updated", exist_ok=True)
os.makedirs("output_720_1024/right_images_updated", exist_ok=True)
os.makedirs("output_720_1024/metadata_updated", exist_ok=True)

# ✅ Define function to save images and metadata
def save_image_and_metadata(idx, left_image, right_image, metadata):
    left_image = left_image.convert("RGB")
    right_image = right_image.convert("RGB")
    
    left_image.save(f"output_720_1024/left_images_updated/left_{idx}.png")
    right_image.save(f"output_720_1024/right_images_updated/right_{idx}.png")

    metadata_path = f"output_720_1024/metadata_updated/meta_{idx}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

# ✅ Use ThreadPoolExecutor for parallel I/O
to_pil_image = ToPILImage()

with ThreadPoolExecutor(max_workers=64) as executor:
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        futures = []
        for i in range(len(batch["left_image"])):
            idx = batch_idx * dataloader.batch_size + i

            # Convert tensors to PIL images
            left_image = to_pil_image(batch["left_image"][i])
            right_image = to_pil_image(batch["right_image"][i])

            # Prepare metadata
            metadata = {
                "item": batch["item"][i],
                "category": batch["category"][i],
                "description_0": batch["description_0"][i],
                "description_1": batch["description_1"][i],
                "collection": batch["collection"][i],
                "quality_assessment": {key: batch["quality_assessment"][key][i].item() for key in batch["quality_assessment"]},
                "target_image_path": f"output_720_1024/left_images_updated/left_{idx}.png",
                "condition_image_path": f"output_720_1024/right_images_updated/right_{idx}.png",
            }

            # Submit the image and metadata saving task to the executor
            futures.append(
                executor.submit(save_image_and_metadata, idx, left_image, right_image,  metadata)
            )

        # Wait for all futures in the batch to complete
        for future in futures:
            future.result()

print("✅ Image and metadata saving completed!")