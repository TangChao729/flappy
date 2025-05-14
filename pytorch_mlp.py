import numpy as np
from torch import nn
from torch import optim
import torch
from typing import List

# Set the global device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class MLPRegression(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = (200, 500, 100),
                 learning_rate: float = 0.001):
        super(MLPRegression, self).__init__()
        layers = []
        for layer_index, dim in enumerate(hidden_dim):
            if layer_index == 0:
                layers.append(nn.Linear(input_dim, dim))
            else:
                layers.append(nn.Linear(hidden_dim[layer_index-1], dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=False))

        self.net = nn.Sequential(*layers).to(DEVICE)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit_step(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray):
        self.train()
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Y = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
        W = torch.tensor(W, dtype=torch.float32, device=DEVICE)

        outputs = self.net(X)
        loss = self.criterion(outputs, Y)
        loss = (loss * W).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            y_pred = self.net(X)
        return y_pred.cpu().numpy()

    def save_model(self, path: str = None):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str = None):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))


if __name__ == '__main__':
    def make_example_dataset(n=1000, dim=4, masking_value=True):
        X = np.random.randn(n, dim)
        y1 = np.max(X, axis=1)
        y2 = np.mean(X, axis=1)
        Y = np.stack([y1, y2], axis=1)
        if masking_value:
            W = (np.random.rand(n, 2) > 0.1) * 1
            Y[W == 0] = -1
        else:
            W = np.ones(Y.shape)
        return X, Y, W

    X, Y, W = make_example_dataset(n=100000, dim=4, masking_value=True)
    X_val, Y_val, _ = make_example_dataset(n=100, dim=4, masking_value=False)
    X_test, Y_test, _ = make_example_dataset(n=100, dim=4, masking_value=False)
    network = MLPRegression(input_dim=4, hidden_dim=[100, 200, 100], output_dim=2)

    iterations = 10000
    batch_size = 100
    for i in range(iterations):
        idx = np.random.permutation(len(X))
        x = X[idx[:batch_size], :]
        y = Y[idx[:batch_size], :]
        w = W[idx[:batch_size], :]
        network.fit_step(x, y, w)

        if i % 100 == 0:
            y_pred = network.predict(X_val)
            print(f'Training iteration #{i}:')
            abs_error = np.mean(np.abs(y_pred[:, 0] - Y_val[:, 0]))
            print('MAE for the 1st value of y: {:0.4f}'.format(abs_error))
            abs_error = np.mean(np.abs(y_pred[:, 1] - Y_val[:, 1]))
            print('MAE for the 2nd value of y: {:0.4f}'.format(abs_error))

    y_pred = network.predict(X_test)
    print(f'Final test set prediction:')
    abs_error = np.mean(np.abs(y_pred[:, 0] - Y_test[:, 0]))
    print('MAE for the 1st value of y: {:0.4f}'.format(abs_error))
    abs_error = np.mean(np.abs(y_pred[:, 1] - Y_test[:, 1]))
    print('MAE for the 2nd value of y: {:0.4f}'.format(abs_error))

    network.save_model(path='temp.ckpt')

    network2 = MLPRegression(input_dim=4, hidden_dim=[100, 200, 100], output_dim=2)
    network2.load_model(path='temp.ckpt')
    y_pred2 = network2.predict(X_test)
    assert np.all(y_pred2 == y_pred)
