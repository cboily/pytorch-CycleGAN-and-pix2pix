# from RadAI.data.pytorch-CycleGAN-and-pix2pix.options.test_options import TestOptions
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from data import UnalignedDataset  # import your custom dataset class

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the saved model from a .pth file
model = torch.load("/../checkpoints/cyclegan_unet_ORL_230406/latest_net_G_A.pth")
model = model.to(device)

# define transforms
transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

# load your unseen data
opt = {}
opt.dataroot = "../../../data/processed/ORL/"
opt.phase = "test"
opt.input_nc = 1
opt.output_nc = 1
opt.max_dataset_size = 1

my_data = UnalignedDataset(opt)
data_loader = DataLoader(my_data, batch_size=1, shuffle=False, num_workers=4)

# evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Accuracy on unseen data: %d %%" % accuracy)


"""class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
# Load numpy array
data = np.load('data.npy')

# Create dataset and dataloader
dataset = NumpyDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through dataloader
for batch_idx, batch_data in enumerate(dataloader):
    # Do something with batch_data (e.g., feed to model)
    print(batch_data.shape)"""
