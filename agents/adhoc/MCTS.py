# original_code_from : https://github.com/saschaschramm/MonteCarloTreeSearch
from baba_is_gym import Env
import numpy as np
from collections import deque
from copy import deepcopy
from util import time_measure
        
def vertical_lines(last_node_flags):
    vertical_lines = []
    vertical_line = '\u2502'
    for last_node_flag in last_node_flags[0:-1]:
        if last_node_flag == False:
            vertical_lines.append(vertical_line + ' ' * 3)
        else:
            # space between vertical lines
            vertical_lines.append(' ' * 4)
    return ''.join(vertical_lines)

def horizontal_line(last_node_flags):
    horizontal_line = '\u251c\u2500\u2500 '
    horizontal_line_end = '\u2514\u2500\u2500 '
    if last_node_flags[-1]:
        return horizontal_line_end
    else:
        return horizontal_line

class Node:
    def __init__(self, action_space, action_margin=0, action=0, won=0, terminal=0):
        self.visited = 0
        self.won = won
        self.child = {}
        self.action = action
        self.untried_actions = list(range(action_margin, action_space+action_margin))
        self.terminal = terminal
        self.reward_sum = 0
        
    def untried_action(self):
        action = np.random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action

    def __str__(self):
        ratio = self.won/self.visited
        return "{}: (visits={}, won={}, ratio={:.2f}, done={})".format(
                                                  self.action,
                                                  self.visited,
                                                  self.won,
                                                  ratio,
                                                  self.terminal
                                                  )

class Monte_carlo_tree_search:
    def __init__(self, enviroment, action_space, action_margin=0):
        self.root = Node(action_space, action_margin)
        self.env = enviroment
        self.action_space = action_space
        self.action_margin = action_margin
        self.node_sequence = deque()
        _, info = self.env.reset()
        self.inner_state = deepcopy(info)
    def select_best_child(self, parent_node, c=np.sqrt(2)):
        max_node = None
        max_utc = -1
        for key, val in parent_node.child.items():
            child_utc = val.won/val.visited + c * np.sqrt(np.log(parent_node.visited)/val.visited)
            if child_utc > max_utc:
                max_utc = child_utc
                max_node = parent_node.child[key]
        return max_node

    def is_expandable(self, node):
        if node.terminal:
            return False
        if len(node.untried_actions) > 0:
            return True
        return False
    
    def add_node(self, new_node, node):
        node.child[str(new_node.action)] = new_node
    
    def expanding(self, node):
        action = node.untried_action()
        _, r, d, _ = self.env.step(action)
        new_node = Node(action=action, action_space=self.action_space, action_margin=self.action_margin, terminal=d)
        self.node_sequence.append(str(new_node.action))
        self.add_node(new_node, node)
        return r, d

    def simulation(self, number_of_steps = 50):
        actions = np.random.randint(0, 5, number_of_steps)
        for action in actions:
            _, r, d, _ = env.step(action)
            if d == 1:
                break

        return r, d

    def backwarding(self, value):
        node = self.root
        node.visited += 1
        node.won += value
        while self.node_sequence:
            node = node.child[self.node_sequence.popleft()]
            node.visited += 1
            node.won += value
            
    def selecting(self):
        node = self.root
        _, _, _, _ = env.step_from_here(self.root.action, self.inner_state)
        while not node.terminal:
            if self.is_expandable(node):
                return self.expanding(node)
            else:
                node = self.select_best_child(node)
                _, r, d, _ = self.env.step(node.action)
                self.node_sequence.append(str(node.action))
        return r, d

    def step(self, steps=10000):
        self.root = Node(self.action_space, self.action_margin)
        for i in range(steps):
            if i % (steps/10) == 0:
                print(i,'step passed')
            reward, done = self.selecting()
            if done :
                if reward < 0:
                    reward = -1
                else:
                    reward = 1
            else:
                reward, done = self.simulation()
                if done == 0:
                    reward = -1
                if reward < 0:
                    reward = -1
                else:
                    reward = 1
            self.backwarding(reward)

    def forward(self):
        action_sequence = deque()
        node = self.root
        while not node.terminal:
            node = self.select_best_child(node, c = 0)
            if node == None:
                break
            action_sequence.append(node.action)
        return action_sequence
    
    def show(self, file_name='MCTS_tree.txt'):
        lines = ""
        for edge, node in self.iter(parent_node=None, depth=0, last_node_flags=[]):
            lines += "{}{}\n".format(edge, node)
        txt_file = open(file_name,'wt', encoding='utf-8')
        txt_file.writelines(lines)
        txt_file.close()
        # print(lines)
    
    def iter(self, parent_node, depth, last_node_flags):
        if parent_node is None:
            parent_node = self.root
        
        if depth == 0:
            yield "", parent_node
        else:
            yield vertical_lines(last_node_flags) + horizontal_line(last_node_flags), parent_node
        
        depth += 1
        for key, child in parent_node.child.items():
            last_node_flags.append(len(child.untried_actions) == self.action_space)
            for edge, node in self.iter(parent_node.child[key], depth, last_node_flags):
                yield edge, node
            last_node_flags.pop()

mode = 'without_interaction'
step_num = 20000
np.random.seed(0)
env = Env(8, training_on_single_stage=True)
mcts = Monte_carlo_tree_search(env, action_space=4, action_margin=1)

if mode == 'without_interaction':
    print('!')
    t = time_measure()
    t.start()
    mcts.step(step_num)
    actions = mcts.forward()
    print('time taken : {:.2f} sec'.format(t.end()))
    print(actions)
    mcts.show()
    _, info = env.reset()
    mcts.inner_state = info
    for action in actions:
        env.render()
        observation, reward, done, info = env.step(action)
        if done == 1:
            print('!!')
else:
    actions = deque()
    _, info = env.reset()
    mcts.inner_state = info
    for i in range(200):
        mcts.step(step_num)
        action = mcts.select_best_child(mcts.root,c=0).action
        actions.append(action)
        observation, reward, done, info = env.step_from_here(action,info)
        # env.render()
        mcts.inner_state = info
        if done == 1:
            break
    print(actions)