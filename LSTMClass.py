import numpy as np

class LSTM:
    
    def __init__(self, hidden_units, input_units, loss, grad_loss, 
                 output_units, learning_rate, output_activation, 
                 max_norm=0.25):
        
        self.hidden_units = hidden_units
        self.input_units = input_units
        self.output_units = output_units
        
        self.eta = learning_rate
        self.output_activation = output_activation
        
        self.loss = loss
        self.grad_loss = grad_loss
        
        self.concat_units = hidden_units + input_units
        self.init_lstm()
        self.max_norm = max_norm
        
    def init_lstm(self):
        
        # weight matrixes for forget, input, output and the external gate
        self.W_f = np.random.randn(self.hidden_units, self.concat_units)
        self.W_i = np.random.randn(self.hidden_units, self.concat_units)
        self.W_g = np.random.randn(self.hidden_units, self.concat_units)
        self.W_o = np.random.randn(self.hidden_units, self.concat_units)
        
        # weight and bias for the output unit: prediction
        self.W_v = np.random.randn(self.output_units, self.hidden_units)
        self.b_v = np.zeros((self.output_units, 1))
        
        # bias vectors for forget, input, output and external gate
        self.b_f = np.zeros((self.hidden_units, 1))
        self.b_i = np.zeros((self.hidden_units, 1))
        self.b_g = np.zeros((self.hidden_units, 1))
        self.b_o = np.zeros((self.hidden_units, 1))
        
        params = [self.W_f, self.W_o, self.W_i, self.W_g, self.W_v]
        new_params = []
        
        for param in params:
            new_param = self.init_orthogonal(param)
            new_params.append(new_param)
        
        [self.W_f, self.W_o, self.W_i, self.W_g, self.W_v] = new_params
        
    def forward(self, inputs):
        
        self.gate_outputs = {'output': {}, 'external': {}, 
                             'input': {}, 'forget': {}, 
                             'candidate': {}}
        
        self.C = {-1: np.zeros((self.hidden_units, 1))}
        self.hidden = {-1: np.zeros((self.hidden_units, 1))}
        
        self.f = {}
        self.i = {}
        self.z = {}
        self.o = {}
        self.g = {}
        
        self.y_outputs = {}
        self.outputs = {}
        
        for t, input_t in enumerate(inputs):
            
            # stack input and hidden states
            self.z[t] = np.row_stack((self.hidden[t-1], input_t))
            self.gate_outputs['external'][t] = self.z[t].copy()
            
            # forget gate
            self.gate_outputs['forget'][t] = np.dot(self.W_f, self.z[t]) + self.b_f
            f = self.sigmoid(self.gate_outputs['forget'][t])
            self.f[t] = f.copy()
            
            # input gate
            self.gate_outputs['input'][t] = np.dot(self.W_i, self.z[t]) + self.b_i
            i = self.sigmoid(self.gate_outputs['input'][t])
            self.i[t] = i.copy()
            
            # candidate gate
            self.gate_outputs['candidate'][t] = np.dot(self.W_g, self.z[t]) + self.b_g
            g = self.tanh(self.gate_outputs['candidate'][t])
            self.g[t] = g.copy()
            
            # cell state
            self.C[t] = f * self.C[t-1] + i * g
            
            # output gate
            self.gate_outputs['output'][t] = np.dot(self.W_o, self.z[t]) + self.b_o
            o = self.sigmoid(self.gate_outputs['output'][t])
            self.o[t] = o.copy()
            
            # hidden state
            self.hidden[t] = o * self.tanh(self.C[t])
            
            # predictions
            self.y_outputs[t] = np.dot(self.W_v, self.hidden[t]) + self.b_v
            output = self.output_activation(self.y_outputs[t])
            self.outputs[t] = output.copy()
            
    def backward(self, targets):
        
        self.dW_f = np.zeros_like(self.W_f)
        self.db_f = np.zeros_like(self.b_f)
        
        self.dW_g = np.zeros_like(self.W_g)
        self.db_g = np.zeros_like(self.b_g)
    
        self.dW_o = np.zeros_like(self.W_o)
        self.db_o = np.zeros_like(self.b_o)
        
        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)
        
        self.dW_v = np.zeros_like(self.W_v)
        self.db_v = np.zeros_like(self.b_v)
        
        self.loss_pass = 0
        
        self.dh_next = np.zeros_like(self.hidden[-1])
        self.dC_next = np.zeros_like(self.hidden[-1])
        
        for t in reversed(range(len(targets))):
            
            # compute loss and extract previous cell state
            self.loss_pass += self.loss(self.outputs[t], targets[t])
            C_prev = self.C[t-1]
            
            # compute grad of the output
            dy = self.grad_loss(self.outputs[t], targets[t])
            
            # update grad for outputs
            self.dW_v += np.dot(dy, self.hidden[t].T)
            self.db_v += dy
            
            # hidden state gradient
            self.dh = np.dot(self.W_v.T, dy)
            
            if len(targets) == 1:
                for t in reversed(range(len(self.z))):
                    self.__backward(t, C_prev)
            else:
                self.__backward(t, C_prev)
            
        grads = [self.dW_f, self.db_f, self.dW_o, self.db_o, 
                 self.dW_i, self.db_i, self.dW_g, self.db_g, 
                 self.dW_v, self.db_v]
    
        [self.dW_f, self.db_f, self.dW_o, self.db_o, 
         self.dW_i, self.db_i, self.dW_g, self.db_g, 
         self.dW_v, self.db_v] = self.clip_gradient_norm(grads, max_norm=self.max_norm)
            
    def optimize(self):
        grads = [self.dW_f, self.db_f, self.dW_o, self.db_o, 
                 self.dW_i, self.db_i, self.dW_g, self.db_g, 
                 self.dW_v, self.db_v]
        
        params = [self.W_f, self.b_f, self.W_o, self.b_o, 
                  self.W_i, self.b_i, self.W_g, self.b_g, 
                  self.W_v, self.b_v]
        
        new_params = []
        for param, grad in zip(params, grads):
            param -= self.eta * grad
            new_params.append(param)
            
        [self.W_f, self.b_f, self.W_o, self.b_o, 
         self.W_i, self.b_i, self.W_g, self.b_g, 
         self.W_v, self.b_v] = new_params
        
    def __backward(self, t, C_prev):
        self.dh += self.dh_next
                    
        # output gate gradient
        self.do = self.dh * self.tanh(self.C[t])
        self.do *= self.sigmoid(self.gate_outputs['output'][t], 
                                derivative=True)
        
        # update grad for output gate
        self.dW_o += np.dot(self.do, self.z[t].T)
        self.db_o += self.do
        
        # hidden cell gradient
        self.dC = np.copy(self.dC_next)
        self.dC += self.dh * self.o[t] * self.tanh(self.C[t], 
                                                derivative=True)
        
        # candidate gate gradient
        self.dg = self.dC * self.i[t]
        self.dg *= self.tanh(self.gate_outputs['candidate'][t], 
                            derivative=True)
        
        # update grad for candidate gate
        self.dW_g += np.dot(self.dg, self.z[t].T)
        self.dW_g += self.dg
        
        # input gate
        self.di = self.dC * self.g[t]
        self.di *= self.sigmoid(self.gate_outputs['input'][t], 
                                derivative=True)
        self.dW_i += np.dot(self.di, self.z[t].T)
        self.db_i += self.di
        
        # forget gate
        self.df = self.dC * C_prev
        self.df *= self.sigmoid(self.gate_outputs['forget'][t], 
                                derivative=True)
        
        self.dW_f += np.dot(self.df, self.z[t].T)
        self.db_f += self.df
        
        dz = (np.dot(self.W_f.T, self.df) + np.dot(self.W_i.T, self.di) + 
            np.dot(self.W_o.T, self.do) + np.dot(self.W_g.T, self.dg))
        
        self.dh_next = dz[:self.hidden_units, :]
        self.dC_next = self.f[t] * self.dC
                    
    @staticmethod
    def tanh(x, derivative=False):
        x += 1e-12
        if not derivative:
            return np.tanh(x)
        return 1.0 - np.tanh(x) ** 2
    
    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1.0 / (1.0 + np.exp(-x))
        if not derivative:
            return sig
        return sig * (1.0 - sig)
    
    @staticmethod
    def clip_gradient_norm(grads, max_norm=0.25):
        # Set the maximum of the norm to be of type float
        max_norm = float(max_norm)
        total_norm = 0

        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for grad in grads:
            grad_norm = np.sum(np.power(grad, 2))
            total_norm += grad_norm

        total_norm = np.sqrt(total_norm)

        # Calculate clipping coeficient
        clip_coef = max_norm / (total_norm + 1e-6)

        # If the total norm is larger than the maximum allowable norm, then clip the gradient
        if clip_coef < 1:
            for grad in grads:
                grad *= clip_coef

        return grads

    @staticmethod
    def init_orthogonal(param):
        if param.ndim < 2:
            raise ValueError("Only parameters with 2 or more dimensions are supported.")

        rows, cols = param.shape

        new_param = np.random.randn(rows, cols)

        if rows < cols:
            new_param = new_param.T

        q, r = np.linalg.qr(new_param)

        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        new_param = q

        return new_param
