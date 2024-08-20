import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch.onnx


class Detector:
    """RTDETR Detector"""

    model_name: str
    detection_threshold: float
    label_dict: dict

    def __init__(
        self: "Detector",
        model_name: str = "PekingU/rtdetr_r18vd",
    ) -> None:
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrForObjectDetection.from_pretrained(model_name)
        self.label_dict = {k: v for k, v in self.model.config.id2label.items()}

    def detect(self, frame, object_name) -> list[list[int]]:
        """Detect object in frame

        Args:
            frame: PIL.Image.Image
            object_name: str

        Returns:
            boxes: list[list[int]] in format [x_min, y_min, x_max, y_max]
        """
        inputs = self.image_processor(images=frame, return_tensors="pt")

        with torch.inference_mode():
            outputs = self.model(**inputs)

            processed_results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([frame.size[::-1]]),
                threshold=0.5,
            )

        print(processed_results)

        boxes = [
            [int(coord) for coord in box.tolist()]
            for result in processed_results
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]
            )
            if self.label_dict[label_id.item()] == object_name
            and score.item() > self.detection_threshold
        ]

        return boxes
