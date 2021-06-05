# requirements

our model is a cnn-lstm deep neural network implemented in pytorch.

run folling code to install requirements:

```
pip install -r requirements.txt
```

# The final model

to load the final model you need 2 files:

1. model wigths `model.pt`
2. leagal tokens `tokens.pickle`

this is how you can load and evaluate them:

```
import pickle
from dataset import CustomDataset
from model import MyModel
import torch
from functions import *

# load a pre-trained model

with open('tokens.pickle', 'rb') as f:
     tokens = pickle.load(f)

model = MyModel(len(tokens) + 1, 32, 128, [256, 128, 64])
model.load_state_dict(torch.load('model.pt'))

# or train a new one

tokens, _, model = generate_model('/path/to/data.train', java_path="C:\\Program Files\\Common Files\\Oracle\\Java\\javapath")
```

the java_path is nessusary dependency from parsivar package. PAS step tagging took about 2 hours on 12core i7 cpu with 100% utilization.

# Accuracy on the test set

```
import pickle
from dataset import CustomDataset
from model import MyModel
import torch
from functions import *

with open('tokens.pickle', 'rb') as file:
    tokens = pickle.load(file)

dataset = CustomDataset('./test.data', tokens=tokens, java_path="C:\\Program Files\\Common Files\\Oracle\\Java\\javapath")

model = MyModel(len(tokens) + 1, 32, 128, [256, 128, 64])
model.load_state_dict(torch.load('model.pt'))

accuracy = compute_accuracy(dataset, model)
print("accuracy on test set: " + str(accuracy))
```

this is eddited on github.
