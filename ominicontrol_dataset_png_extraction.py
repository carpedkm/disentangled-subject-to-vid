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
dataset = load_dataset("Yuanshi/Subjects200K")

def filter_func(item):
    if not item.get("quality_assessment"):
        return False
    return all(
        item["quality_assessment"].get(key, 0) >= 5
        for key in ["compositeStructure", "objectConsistency", "imageQuality"]
    )

data_valid = dataset["train"].filter(
    filter_func, num_proc=16, cache_file_name="./cache/dataset/data_valid.arrow"
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
        left_img = image.crop((0, 0, 512, 512)).resize((512, 512)).convert("RGB")
        right_img = image.crop((512, 0, 1024, 512)).resize((512, 512)).convert("RGB")

        description_0 = item["description"]["description_0"]
        description_1 = item["description"]["description_1"]

        return {
            "left_image": self.to_tensor(left_img),
            "right_image": self.to_tensor(right_img),
            "description_0": description_0,
            "description_1": description_1,
            "collection": item["collection"],
            "quality_assessment": item["quality_assessment"],
        }

subject_dataset = Subject200KDataset(data_valid)

# ✅ Use DataLoader for batch processing
dataloader = DataLoader(subject_dataset, batch_size=32, num_workers=8)

# ✅ Create output directories
os.makedirs("output/left_images", exist_ok=True)
os.makedirs("output/right_images", exist_ok=True)
os.makedirs("output/metadata", exist_ok=True)

# ✅ Define function to save images and metadata
def save_image_and_metadata(idx, left_image, right_image, metadata):
    left_image.save(f"output/left_images/left_{idx}.png")
    right_image.save(f"output/right_images/right_{idx}.png")

    metadata_path = f"output/metadata/meta_{idx}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

# ✅ Use ThreadPoolExecutor for parallel I/O
to_pil_image = ToPILImage()

with ThreadPoolExecutor(max_workers=8) as executor:
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        futures = []
        for i in range(len(batch["left_image"])):
            idx = batch_idx * dataloader.batch_size + i

            # Convert tensors to PIL images
            left_image = to_pil_image(batch["left_image"][i])
            right_image = to_pil_image(batch["right_image"][i])

            # Prepare metadata
            metadata = {
                "description_0": batch["description_0"][i],
                "description_1": batch["description_1"][i],
                "collection": batch["collection"][i],
                "quality_assessment": {key: batch["quality_assessment"][key][i].item() for key in batch["quality_assessment"]},
                "target_image_path": f"output/left_images/left_{idx}.png",
                "condition_image_path": f"output/right_images/right_{idx}.png",
            }

            # Submit the image and metadata saving task to the executor
            futures.append(
                executor.submit(save_image_and_metadata, idx, left_image, right_image, metadata)
            )

        # Wait for all futures in the batch to complete
        for future in futures:
            future.result()

print("✅ Image and metadata saving completed!")