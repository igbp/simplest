import torch
import numpy as np
import matplotlib.pyplot as plt

class World():
    def __init__(self):
        self.cols = 10
        self.rows = 10
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
                self.current = (self.current[0], self.current[1]-1)

        if self.current == self.end:
            self.done = True

        return self.current, self.done

class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.actions = ['N', 'S', 'E', 'W']
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

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
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
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



    def print_action(self, world):
        ncols = world.cols
        nrows = world.rows
        print(f'------------------')
        grid = [[0 for x in range(ncols)] for y in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                obs = torch.tensor([i, j], dtype=torch.float32)
                action = self.actor(obs)
                grid[i][j] = self.actions[action.argmax()]

        [print(grid[row]) for row in range(nrows)]


def gen_expert_data():

    w_expert = World()
    expert_data_set = {}

    for j in range(0, w_expert.cols):
        expert_data_set[(0, j)] = 'W'

    for i in range(0, w_expert.rows):
        expert_data_set[(i, w_expert.cols - 1)] = 'S'

    for j in reversed(range(0, w_expert.cols)):
        expert_data_set[(w_expert.rows - 1, j)] = 'E'

    for i in reversed(range(1, w_expert.rows)):
        expert_data_set[(i, 0)] = 'N'

    grid ={}
    for i in range(0, w_expert.rows):
        for j in range(0, w_expert.cols):
            if (i, j) not in expert_data_set.keys():
                grid[(i, j)] = 'X'
            else:
                grid[(i, j)] = expert_data_set[(i, j)]

    grid_l = [[grid[(i, j)] for j in range(0, w_expert.cols)] for i in range(0, w_expert.rows)]
    [print(grid_l[row]) for row in range(w_expert.rows)]

    return expert_data_set


def compare_agreement():


def main():
    w = World()
    _, _ = w.reset()
    expert_data = gen_expert_data()
    print(expert_data)
    agent_a = Agent()
    agent_a.train(expert_data)
    agent_a.print_action(w)

    agent_b = Agent()
    agent_b.train(expert_data)
    agent_b.print_action(w)


if __name__ == '__main__':
    main()


