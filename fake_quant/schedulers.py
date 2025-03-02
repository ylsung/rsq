import torch
import yaml


class Scheduler:
    def __init__(self):
        pass
    
    def get_schedule(self, max_length):
        pass
    
    def normalize_weight(self, input_tensor, min_value, max_value, quantile_value=None):
        if quantile_value is not None:
            q_min = 1 - quantile_value
            q_max = quantile_value
            # Ensure that q_max is the larger quantile
            if q_max < q_min:
                q_max, q_min = q_min, q_max
            quantile = torch.tensor([q_min, q_max]).to(input_tensor.device)
            tensor_min, tensor_max = torch.quantile(input_tensor, quantile)
        else: 
            tensor_min, tensor_max = torch.min(input_tensor), torch.max(input_tensor)
        normalized_input = (input_tensor - tensor_min) / (tensor_max - tensor_min)
        normalized_input = normalized_input * (max_value - min_value) + min_value
        normalized_input.clamp_(min_value, max_value)
        return normalized_input

    def get_ratio(self, max_length):
        schedule = self.get_schedule(max_length)
        schedule = schedule[torch.arange(max_length)]

        return self.normalize_weight(schedule, min_value=self.min_value, max_value=self.max_value)
    

class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, **kwargs):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        
        self.min_value = min(start_value, end_value)
        self.max_value = max(start_value, end_value)
        
    def get_schedule(self, max_length):
        return torch.linspace(self.start_value, self.end_value, max_length)
    

class EndPointsPeakScheduler(Scheduler):
    def __init__(self, min_value, max_value, start_value=1, end_value=1, factor=6, **kwargs):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.min_value = min_value
        self.max_value = max_value
        self.factor = factor
        
    def get_schedule(self, max_length):
        # Generate x values between start and end
        start = 0
        end = max_length - 1
        x = torch.linspace(start, end, max_length)
        
        # Generate y values for the curve using a cosine function raised to a power
        y = torch.cos((x - start) * torch.pi / (end - start))**self.factor
        
        return y
    

class StartPeakScheduler(Scheduler):
    def __init__(self, min_value, max_value, start_value=1, end_value=1, factor=6, **kwargs):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.min_value = min_value
        self.max_value = max_value
        self.factor = factor
        
    def get_schedule(self, max_length):
        # Generate x values between start and end
        start = 0
        end = max_length - 1
        x = torch.linspace(start, end, max_length)
    
        # Shift x values to avoid division by zero at start
        y = 1 / ((x - start + 1)**self.factor)
        
        return y


def load_scheduler(yaml_file_path, **kwargs):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    method_name = config['method_name']
    params = config['params']
    
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params.update(**kwargs)
    
    try:
        return eval(method_name)(**params)

    except NameError:
        raise ValueError(f"Unknown scheduler {method_name}")