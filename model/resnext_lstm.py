import torch
from torch import nn
from torchvision import models


model = models.resnext50_32x4d(pretrained=False)  # Residual Network CNN


class ResNext_LSTM(nn.Module):
    def __init__(self, num_class, image_size=224, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(ResNext_LSTM, self).__init__()
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.image_size = image_size
        self.hidden_size = hidden_dim
        self.latent_size = latent_dim
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(self.hidden_size, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(-1, batch_size, self.latent_size)
        x_lstm, _ = self.lstm(x, None)
        return self.dp(self.linear1(x_lstm.view(batch_size,self.hidden_size)))


if __name__ == '__main__':
    inp = torch.randn(128, 3, 320, 320)
    cnn_rnn = ResNext_LSTM(2, latent_dim=2048, lstm_layers=3,
                        hidden_dim=256, bidirectional=False)
    out = cnn_rnn(inp)
