import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Define image embedding
        self.image_embedding_size = 64
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(2, 2)),
            nn.ReLU()
        )

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class AdditionEncoderRNN(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, device=None):
        super(AdditionEncoderRNN, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hidden):
        # input has to be of shape (seq_len=19, batch, input_size=10)
        # hidden is a tuple with both elements of shape (1, batch, hidden_size=128)
        output, hidden = self.lstm(input, hidden)
        return output, hidden

    def initHidden(self, batch_size=4096):
        return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, batch_size, self.hidden_size, device=self.device))


class AdditionDecoderRNN(nn.Module):
    def __init__(self, hidden_size=128, output_size=10, seq_len=10, device=None):
        super(AdditionDecoderRNN, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.seq_len = seq_len

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        # input should be the encoder's output[-1] of shape (batch_size, hidden_size)
        output = input.repeat(self.seq_len, 1, 1)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output))
        # output should be of shape (seq_len=10, batch_size, output_size=10) with values being log-probabilities
        return output

    def initHidden(self, batch_size=4096):
        return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, batch_size, self.hidden_size, device=self.device))


class AdditionModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, output_size=10, output_seq_len=10, device=None):
        super(AdditionModel, self).__init__()
        self.encoder = AdditionEncoderRNN(input_size, hidden_size, device)
        self.decoder = AdditionDecoderRNN(hidden_size, output_size, output_seq_len, device)

    def forward(self, input):
        # input has to be of shape (seq_len=19, batch, input_size=11)
        batch_size = input.size(1)
        hidden = self.encoder.initHidden(batch_size)

        encoder_output, _ = self.encoder(input, hidden)
        decoder_input = encoder_output[-1]

        hidden = self.decoder.initHidden(batch_size)
        decoder_output = self.decoder(decoder_input, hidden)

        # decoder_output is of shape(seq_len=10, batch, output_size=10)
        return decoder_output
