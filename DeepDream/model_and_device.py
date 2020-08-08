from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
