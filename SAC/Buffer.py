import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, seq_len):
        self.max_size = max_size
        self.seq_len = seq_len
        self.mem_cntr = 0
        self.states = np.zeros((max_size, input_shape))
        self.actions = np.zeros((max_size, n_actions))
        self.rewards = np.zeros(max_size)
        self.next_states = np.zeros((max_size, input_shape))
        self.terminals = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.max_size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.max_size)

        idxs = np.random.randint(0, max_mem - self.seq_len, size=batch_size)

        # Check for consecutive 'done' flags, and discard the sequence if it crosses episode boundaries
        while True:
            done_flags = np.array([self.terminals[i:i+self.seq_len-1] for i in idxs])
            invalid_idxs = np.any(done_flags, axis=1)
            if not np.any(invalid_idxs):
                break
            idxs[invalid_idxs] = np.random.randint(0, max_mem - self.seq_len, size=np.sum(invalid_idxs))
            
        states = np.array([self.states[i:i+self.seq_len] for i in idxs])
        actions = np.array([self.actions[i:i+self.seq_len] for i in idxs])
        rewards = np.array([self.rewards[i+self.seq_len-1] for i in idxs])
        next_states = np.array([self.next_states[i:i+self.seq_len] for i in idxs])
        terminals = np.array([self.terminals[i+self.seq_len-1] for i in idxs])

        return states, actions, rewards, next_states, terminals

class ReplayBuffer_():
    def __init__(self, max_size, input_shape, n_actions, seq_len):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, seq_len, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, seq_len, input_shape))
        self.action_seq_memory = np.zeros((self.mem_size, seq_len, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action_seq, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_seq_memory[index] = action_seq
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        action_seq = self.action_seq_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, action_seq, rewards, states_, dones
    