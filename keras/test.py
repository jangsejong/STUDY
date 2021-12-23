class Calc:
    def __init__(self, n1, n2):
        self.n1 = n1 
        self.n2 = n2
        return print(self.n1, self.n2)

    def __call__(self, n1, n2):
        self.n1 = n1 
        self.n2 = n2
        return print(self.n1 + self.n2)

#s = Calc(1,2)

#print(s)
Calc.__call__(1, 2)
Calc(2,3)