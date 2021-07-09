from datasets import create_dataloader
from options import create_options
from models import create_model

# handle options
opt = create_options()
opt.phase = 'train'

# Create classes
training_data, training_dataset = create_dataloader(opt, 'train')
validation_data, validation_dataset = create_dataloader(opt, 'val')
model = create_model(opt, training_dataset)

model.run_experiment(training_data, validation_data, None)
