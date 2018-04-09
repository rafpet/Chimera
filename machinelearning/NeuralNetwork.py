import random
import math

class Matrix:
    
    def __init__(self,_rows, _columns):
        
        self.rows = _rows
        self.columns = _columns
        self.data = []
        
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                row.append(0)
            self.data.append(row)
        
    @staticmethod
    def from_array(_array):
        
        m = Matrix(len(_array),1)
        
        for i in range(len(_array)):
            m.data[i][0] = _array[i]
        
        return m
        
    def to_array(self):
        
        array = []
        
        for i in range(self.rows):
            for j in range(self.columns):
                array.append(self.data[i][j])
                
        return array
        
    def scale(self, _n):
        
        if isinstance(_n, Matrix):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.data[i][j] *= _n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.data[i][j] *= _n
                    

    def static_map(_matrix, _func):
        
        output = Matrix(_matrix.rows, _matrix.columns)
        for i in range(_matrix.rows):
            for j in range(_matrix.columns):
                value = _matrix.data[i][j]
                output.data[i][j] = _func(value)
                
        return output
    
    def map(self, _func):
        
        for i in range(self.rows):
            for j in range(self.columns):
                value = self.data[i][j]
                self.data[i][j] = _func(value)
    
    def subtract(_a, _b):
        
        output = Matrix(_a.rows, _a.columns)
        
        for i in range(output.rows):
            for j in range(output.columns):
                output.data[i][j] = _a.data[i][j] - _b.data[i][j]
                
        return output
   
    def add(self, _n):
        
        if isinstance(_n, Matrix):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.data[i][j] += _n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.data[i][j] += _n
    
    def static_multiply(_a,_b):
        
        if isinstance(_a, Matrix):
            
            if _a.columns != _b.rows:
                print "ERROR: columns of a must match rows of b"
                
            else:
                
                result = Matrix(_a.rows, _b.columns)
                
                for i in range(result.rows):
                    for j in range(result.columns):
                        sum = 0
                        for k in range(_a.columns):
                            sum += _a.data[i][k] * _b.data[k][j] 
                        
                        result.data[i][j] = sum
                        
                return result
            
    def multiply(self, _n):
        
        if isinstance(_n, Matrix):
            
            if self.columns != _n.rows:
                print "ERROR: columns of a must match rows of b"
                
            else:
                
                result = Matrix(self.rows, _n.columns)
                
                for i in range(result.rows):
                    for j in range(result.columns):
                        sum = 0
                        for k in range(self.columns):
                            sum += self.data[i][k] * _n.data[k][j] 
                        
                        result.data[i][j] = sum
                        
                return result
        else:
            pass
            
            
    @staticmethod
    def transpose(_m):
        result = Matrix(_m.columns,_m.rows)
        for i in range(_m.rows):
            for j in range(_m.columns):
                result.data[j][i] = _m.data[i][j]
        return result
        
    def randomise(self):
        for i in range(self.rows):
            for j in range(self.columns):
                self.data[i][j] = random.uniform(-1,1)
                
    def show(self):
        print ""
        for x in self.data:
            print x
        print ""

"""
class ActivationFunction:
    
    def __init__(self, 
    
"""
    
class NeuralNetwork:
    
    def __init__(self, _num_inputs, _num_hidden, _num_outputs):
        
        self.input_nodes = _num_inputs
        self.hidden_nodes =_num_hidden
        self.output_nodes = _num_outputs
        
        self.weights_ih = Matrix(self.hidden_nodes,self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes,self.hidden_nodes)
        
        self.weights_ih.randomise()
        self.weights_ho.randomise()
        
        self.bias_h = Matrix(self.hidden_nodes,1)
        self.bias_o = Matrix(self.output_nodes,1)
        
        self.bias_h.randomise()
        self.bias_o.randomise()
        
        self.learning_rate = .1
        
    def predict(self, _input_array):
        
        inputs = Matrix.from_array(_input_array)
        
        #layer 1
        hidden = Matrix.static_multiply(self.weights_ih,inputs)
        hidden.add(self.bias_h)
        
        hidden.map(sigmoid)
        
        #layer 2
        output = Matrix.static_multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        
        output.map(sigmoid)
        
        return output.to_array()
        
    def train(self, _input_array, _targets_array):
        
        #generating hidden outputs
        inputs = Matrix.from_array(_input_array)
        
        hidden = Matrix.static_multiply(self.weights_ih,inputs)
        hidden.add(self.bias_h)
        #activation function
        hidden.map(sigmoid)
        
        #generating output
        outputs = Matrix.static_multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)
        
        #Calculate errors
        #Error = Targets - Outputs
        targets = Matrix.from_array(_targets_array)
        output_errors = Matrix.subtract(targets, outputs)
        
        #outputs.show()
        #targets.show()
        #output_errors.show()
        
        gradients = Matrix.static_map(outputs,dsigmoid)
        gradients.scale(output_errors)
        gradients.scale(self.learning_rate)
        
        #calculate deltas
        hidden_t = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.static_multiply(gradients, hidden_t)
        #adjust the weights by deltas
        self.weights_ho.add(weights_ho_deltas)
        #adjust the bias by its deltas
        self.bias_o.add(gradients)
        
        #Calculate the hidden layer errors
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.static_multiply(weights_ho_t,output_errors)
        
        #calculate hidden gradient
        hidden_gradient = Matrix.static_map(hidden, dsigmoid)
        #hidden_gradient.show
        #hidden_errors.show()
        #print ""
        hidden_gradient.scale(hidden_errors)
        hidden_gradient.scale(self.learning_rate)
        
        #calculate input -> hidden deltas
        inputs_t = Matrix.transpose(inputs)
        #inputs_t.show()
        #hidden_gradient.show()
        #print ""
        weights_ih_deltas = Matrix.static_multiply(hidden_gradient, inputs_t)
        #adjust the bias by its deltas
        self.bias_h.add(hidden_gradient)
        
        self.weights_ih.add(weights_ih_deltas)
        
        
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def dsigmoid(y):
    #return sigmoid(x) * (1 - sigmoid(x))
    return y * (1 - y)
    
    
    
    
