from .base import DatasetGenerator
from prompts.alea_prompt import create_prompt


class RandomPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
        augmentor=None,
    ):
        super().__init__(generator, batch_size, output_dir, augmentor)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}
        for label in labels_names:
            prompts[label] = []
            for i in range(self.num_images_per_label):
                prompts[label].append(
                    {
                        "prompt": create_prompt(label),
                        "num_images": 1,
                    }
            )
        return prompts
