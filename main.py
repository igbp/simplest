import torch
import numpy as np
import matplotlib.pyplot as plt

class World():
    def __init__(self):
        self.cols = 4
        self.rows = 5
        self.start = (0, 0)
        self.end = (1, 0)
        self.valid_actions = ['N', 'S', 'E', 'W']

    def print_world(self):
        print(f'---------------')
        self.grid = [[0 for x in range(self.cols)] for y in range(self.rows)]
        self.grid[self.current[0]][self.current[1]] = 1
        for row in self.grid:
            print(row)
        print(f'Current position: {self.current}')
        print(f'done = {self.done}')

    def reset(self):
        self.grid = [[0 for x in range(self.cols)] for y in range(self.rows)]
        self.current = (0, 0)
        self.grid[self.current[0]][self.current[1]] = 1
        self.done = False
        return self.current, self.done

    def step(self, action):
        if action not in self.valid_actions:
            raise Exception('Invalid action')

        if action == 'N':
            if self.current[0] == 0:
                self.current = (self.current[0], self.current[1])
            else:
                self.current = (self.current[0]-1, self.current[1])

        if action == 'S':
            if self.current[0] == self.rows-1:
                self.current = (self.current[0], self.current[1])
            else:
                self.current = (self.current[0] + 1, self.current[1])

        if action == 'W':
            if self.current[1] == self.cols-1:
                self.current = (self.current[0], self.current[1])
            else:
                self.current = (self.current[0], self.current[1]+1)

        if action == 'E':
            if self.current[1] == 0:
                self.current = (self.current[0], self.current[1])
            else:
                self.current = (self.current[0], self.current[1] -1)

        if self.current == self.end:
            self.done = True

        return self.current, self.done

class Agent():
    def __init__(self):
        self.actions = ['N', 'S', 'E', 'W']
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
            torch.nn.Softmax(dim=0))

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = self.actor(obs)

        return action

    def print_weights(self):
        print(self.actor[0].weight)
        print(self.actor[0].bias)
        print(self.actor[2].weight)
        print(self.actor[2].bias)

    def convert_action(self, action):
        acs_vector = np.zeros(len(self.actions))
        acs_vector[self.actions.index(action)] = 1
        return torch.tensor(acs_vector, dtype=torch.float32)

    def train(self, expert_data):
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_history = []
        for i in range(1000):
            loss = 0
            for obs, action in expert_data.items():
                optimizer.zero_grad()
                action_vector = self.convert_action(action)
                pred_action = self.get_action(obs)
                loss += loss_fn(pred_action, action_vector)
            loss_history.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        plt.plot(loss_history)
        plt.show()
        torch.save(self.actor, 'model.pth')



    def print_action(self):
        print(f'------------------')
        grid = [[0 for x in range(4)] for y in range(5)]
        for i in range(5):
            for j in range(4):
                obs = torch.tensor([i, j], dtype=torch.float32)
                action = self.actor(obs)
                grid[i][j] = self.actions[action.argmax()]

        [print(grid[row]) for row in range(5)]
]

if __name__ == '__main__':
    w = World()
    obs, done = w.reset()
    expert_data = {(0,0):'W', (0,1):'W', (0,2):'W',  (0,3):'S',
                   (1,3):'S', (2,3):'S', (3,3):'S',  (4,3):'E',
                   (4,2):'E', (4,1):'E', (4,0):'N',
                   (3,0):'N', (2,0):'N', (1,0):'N'}


    agent1 = Agent()
    agent2 = Agent()
    agent3 = Agent()

    agent1.train(expert_data)
    agent2.train(expert_data)
    agent3.train(expert_data)

    agent1.print_action()
    agent2.print_action()
    agent3.print_action()



