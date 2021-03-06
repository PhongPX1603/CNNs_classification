from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

import utils


def chucks(lst: List, size: Optional[int] = None):
	if size is None:
		yield lst
	else:
		for i in range(0, len(lst), size):
			yield lst[i:i + size]

class Predictor(nn.Module):
	def __init__(self, 
				 model_config: Dict = None,
				 classes: Dict = None,
				 weight_path: Optional[str] = None,
				 image_size: Optional[int] = None,
				 mean: Optional[Tuple[float, float, float]] = None,
				 std: Optional[Tuple[float, float, float]] = None,
				 batch_size: Optional[int] = None,
				 device: str = 'cpu') -> None:
		super(Predictor, self).__init__()
		self.classes = classes
		self.image_size = image_size
		self.batch_size = batch_size
		self.device = device

		self.mean = torch.tensor(mean, dtype=torch.float, device=device).view(1, 3, 1, 1) if mean else None
		self.std = torch.tensor(std, dtype=torch.float, device=device).view(1, 3, 1, 1)	if std else None

		self.model = utils.create_instance(model_config)

		if weight_path is not None:
			self.model.load_state_dict(torch.load(f=weight_path, map_location='cpu'))
		self.model.to(self.device).eval()

	def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
		images = [cv2.resize(image, dsize=self.image_size) for image in images]
		if (self.mean is not None) and (self.std is not None):
			images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
		return images

	def process(self, images: List[np.ndarray]) -> List[torch.Tensor]:
		preds = []
		for batch in chucks(images, self.batch_size):
			batch = [torch.from_numpy(image) for image in batch]
			batch = torch.stack(batch, dim=0).to(self.device)
			batch = batch.permute(0, 3, 1, 2).contiguous().float()
			if (self.mean is not None) and (self.std is not None):
				batch = (batch.div(255.) - self.mean) / self.std
			else:
				batch = (batch - batch.mean()) / batch.std()
			with torch.no_grad():
				preds += torch.split(self.model(batch), split_size_or_sections=1, dim=0)

		return preds

	def postprocess(self, preds: List[torch.Tensor]) -> List[Tuple[str, float]]:
		outputs = []

		for pred in preds:
			pred = pred.softmax(dim=1).squeeze(dim=0)
			scores = pred.data.cpu().numpy().tolist()
			class_score = max(scores)
			class_name = self.classes[scores.index(class_score)]
			outputs.append((class_name, class_score))

		return outputs

	def forward(self, images):
		images = self.preprocess(images)
		preds = self.process(images)
		outputs = self.postprocess(preds)

		return outputs