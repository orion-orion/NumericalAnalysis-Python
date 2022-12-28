import math

class Var:
    def __init__(self, val, deriv=1.0):
        self.val = val
        self.deriv = deriv
    
    def __add__(self, other):
        if isinstance(other, Var):
            val = self.val + other.val
            deriv = self.deriv + other.deriv
        else:
            val = self.val + other
            deriv = self.deriv
        return Var(val, deriv)
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Var):
            val = self.val - other.val
            deriv = self.deriv - other.deriv
        else:
            val = self.val - other
            deriv = self.deriv
        return Var(val, deriv)
    
    def __rsub__(self, other):
        val = other - self.val
        deriv = - self.deriv
        return Var(val, deriv)

    def __mul__(self, other):
        if isinstance(other, Var):
            val = self.val * other.val
            deriv = self.val * other.deriv + self.deriv * other.val
        else:
            val = self.val * other
            deriv = self.deriv * other
        return Var(val, deriv)
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Var):
            val = self.val / other.val
            deriv = (self.deriv * other.val - self.val * other.deriv)/other.val**2
        else:
            val = self.val / other
            deriv = self.deriv / other
        return Var(val, deriv)

    def __rtruediv__(self, other):
        val = other / self.val
        deriv = other * 1/self.val**2
        return Var(val, deriv)
    
    def __repr__(self):
        return "value: {}\t deriv: {}".format(self.val, self.deriv)
        

def exp(f: Var):
    return Var(math.exp(f.val), math.exp(f.val) * f.deriv)


fx = lambda x: exp(x*x - x)/x

df = fx(Var(2.0))
print(df)

# value: 3.694528049465325         deriv: 9.236320123663312
