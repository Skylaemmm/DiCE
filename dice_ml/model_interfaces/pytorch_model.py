"""Module containing an interface to trained PyTorch model."""

from dice_ml.model_interfaces.base_model import BaseModel
import torch

class PyTorchModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='PYT'):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        :param backend: "PYT" for PyTorch framework.
        """

        super().__init__(model, model_path, backend)

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_tensor):
        return self.model(input_tensor).float()

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")


class GPPyTorchModel(BaseModel):

    def __init__(self, model=None, likelihood = None, model_path='', backend='GPPYT'):
        """Init method

        :param model: trained PyTorch Model.
        :param likelihood: trained likelihood.
        :param model_path: path to trained model.
        :param backend: "PYT" for PyTorch framework.
        """

        super().__init__(model, likelihood, model_path, backend)

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_tensor):
        input_shape_list = list(input_tensor.shape)
        if len(input_shape_list) ==1:
            input_tensor = input_tensor.reshape(1,input_shape_list[0])
        return self.likelihood(self.model(input_tensor)).mean.ge(0.5).float()

    def set_eval_mode(self):
        self.model.eval()
        self.likelihood.eval()

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")
