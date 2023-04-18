import torch

class world():
    def __init__(self):
        self.cols = 4
        self.rows = 3
        self.start = (0, 0)
        self.end = (self.rows-1, self.cols-1)
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





if __name__ == '__main__':
    w = world()
    obs, done = w.reset()
    w.print_world()

    obs, done = w.step('W')
    w.print_world()

    obs, done = w.step('W')
    w.print_world()

    obs, done = w.step('W')
    w.print_world()

    obs, done = w.step('S')
    w.print_world()

    obs, done = w.step('S')
    w.print_world()

