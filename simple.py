
class Operator:

    def __init__(self, kind):
        self.kind = kind

    def add(self, a, b):
        self.a = a
        self.b = b
        out = Variable(a.val + b.val, trace=self)
        self.out = out
        return out

    def mul(self, a, b):
        self.a = a
        self.b = b
        out = Variable(a.val * b.val, trace=self)
        self.out = out
        return out
    
    def backward(self, upstream):
        if self.kind == "add":
            a_grad = upstream
            b_grad = upstream
            self.a.grad += a_grad
            self.b.grad += b_grad

            self.a.backward(a_grad)
            self.b.backward(b_grad)

        elif self.kind == "mul":
            a_grad = upstream * self.b.val
            b_grad = upstream * self.a.val
            self.a.grad += a_grad
            self.b.grad += b_grad

            self.a.backward(a_grad)
            self.b.backward(b_grad)


class Variable:

    def __init__(self, val, trace=None):
        self.val = val
        self.trace = trace
        self.grad = 0

    def __mul__(self, other):
        op = Operator("mul")
        return op.mul(self, other)

    def __add__(self, other):
        op = Operator("add")
        return op.add(self, other)

    def backward(self, upstream=1):
        if self.trace is not None:
            self.trace.backward(upstream)

    

x = Variable(2)
y = Variable(3)

z = x * (x + y) + y*y 

# expert 16
print(f'{z.val=}')

z.backward()

# expert 7, 8
print(x.grad)
print(y.grad)