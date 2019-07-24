import torch


class SLAlgo:
    """A supervised learning algorithm."""

    def __init__(self, gen, model, criterion, lr=0.001, adam_eps=1e-8,
                 batch_size=256, num_batches=10, eval_num_examples=100):
        self.gen = gen
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.eval_num_examples = eval_num_examples

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, eps=adam_eps)
        self.num_examples = self.batch_size * self.num_batches

    def generate_data(self):
        X, Y = self.gen.generate(self.num_examples)

        logs = {
            "num_examples": self.num_examples
        }

        return (X, Y), logs

    def update_parameters(self, X, Y):
        loss = 0

        batch_start_inds = range(0, len(X), self.batch_size)
        num_batches = len(batch_start_inds)
        for i in batch_start_inds:
            batch_X = X[i:i+self.batch_size]
            batch_Y = Y[i:i+self.batch_size]

            batch_loss = self.criterion(self.model(batch_X), batch_Y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()

        loss /= num_batches

        logs = {
            "loss": loss
        }

        return logs

    def evaluate(self):
        return self.gen.evaluate(self.model, self.eval_num_examples)
