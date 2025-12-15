# Unit tests for models using pytest.

import torch
import pytest
from hogat.models.hogat_model import HOGAT
from torch_geometric.data import Data

@pytest.fixture
def sample_data():
    x = torch.rand(10, 128)
    edge_index = torch.tensor([[0,1,2],[1,2,0]])
    return Data(x=x, edge_index=edge_index)

def test_hogat_forward(sample_data):
    model = HOGAT(input_dim=128, hidden_dim=64, output_dim=5)
    out = model(sample_data)
    assert out.shape == (10, 5) 
    assert torch.all((out >= 0) & (out <= 1)) 
