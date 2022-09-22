"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import minitorch
import random


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        # ASSIGN1.5
        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)
        # END ASSIGN1.5

    def forward(self, x):
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        # ASSIGN1.5
        # print(f"self ",end=" :: ")
        # print([i.value.unique_id for i in self.parameters()])
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y
        # END ASSIGN1.5


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        print("+" * 40)
        print(self.model)
        print(len(self.model.layer1.parameters()))
        print(len(self.model.layer2.parameters()))
        print(len(self.model.layer3.parameters()))
        print(self.model.parameters(), len(self.model.parameters()))
        print([i.value.unique_id for i in self.model.parameters()])

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                total_loss += loss.data

            losses.append(total_loss)

            # print("-"*40)
            # print([i.value.unique_id for i in self.model.layer1.parameters()])
            # print(len(self.model.parameters()),[i.value.unique_id for i in self.model.parameters()])
            # print("-"*40)

            # Update
            optim.step()

            # update will re-create new Variables; so need to re-register
            optim = minitorch.SGD(self.model.parameters(), learning_rate)

            # print("-"*40)
            # print([i.value.unique_id for i in self.model.layer1.parameters()])
            # print(len(self.model.parameters()),[i.value.unique_id for i in self.model.parameters()])
            # print(self.model.parameters(), len(self.model.parameters()))
            # print("-"*40)
            # raise

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Diag"](PTS)
    # data = minitorch.datasets["Simple"](PTS)

    # update epoch
    max_epochs=5000
    RATE = 1e-4 * RATE
    ScalarTrain(HIDDEN).train(data, RATE, max_epochs)
