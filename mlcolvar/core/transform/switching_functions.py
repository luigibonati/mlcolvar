import torch

from mlcolvar.core.transform import Transform


__all__ = ["SwitchingFunctions"]

class SwitchingFunctions(Transform):
    """
    Transform class with some common switching functions
    """
    SWITCH_FUNCS = ['Fermi', 'Rational']

    def __init__(self,
                 in_features : int,
                 name : str, 
                 cutoff : float, 
                 options : dict = None):
        f"""Initialize switching function object

        Parameters
        ----------
        name : str
            Name of the switching function to be used, available {",".join(self.SWITCH_FUNCS)}
        cutoff : float
            Cutoff for the swtiching functions
        options : dict, optional
            Dictionary with all the arguments of the switching function, by default None
        """
        super().__init__(in_features=in_features, out_features=in_features)

        self.name = name
        self.cutoff = cutoff
        if options is None:
            options = {}
        self.options = options
       
        if name not in self.SWITCH_FUNCS:
            raise NotImplementedError(f'''The switching function {name} is not implemented in this class. The available options are: {",".join(self.SWITCH_FUNCS)}.
                    You can initialize it as a method of the SwitchingFunctions class and tell us on Github, contributions are welcome!''')  

    def forward(self, x : torch.Tensor):
        switch_function = getattr(self, f'{self.name}_switch')
        y = switch_function(x, self.cutoff, **self.options)
        return y
    
    # ========================== define here switching functions ==========================
    def Fermi_switch(self,
                     x : torch.Tensor, 
                     cutoff : float, 
                     q : float = 0.01, 
                     prefactor_cutoff : float = 1.0):
        y = torch.div( 1, ( 1 + torch.exp( torch.div((x - prefactor_cutoff*cutoff) , q ))))
        return y

    def Rational_switch(self,
                        x : torch.Tensor, 
                        cutoff : float, 
                        n : int = 6, 
                        m : int = 12, 
                        eps : float = 1e-8, 
                        prefactor_cutoff : float = 1.0):
        y = torch.div((1 - torch.pow(x/(prefactor_cutoff*cutoff), n) + eps) , (1 - torch.pow(x/(prefactor_cutoff*cutoff), m)  + 2*eps) )
        return y


def test_switchingfunctions():
    x = torch.Tensor([1., 2., 3.])
    cutoff = 2
    switch = SwitchingFunctions(in_features=len(x), name='Fermi', cutoff=cutoff)
    out = switch(x)

    switch = SwitchingFunctions(in_features=len(x), name='Fermi', cutoff=cutoff, options = {'q' : 0.5})
    out = switch(x)

    switch = SwitchingFunctions(in_features=len(x), name='Rational', cutoff=cutoff, options = {'n' : 6, 'm' : 12})
    out = switch(x)

if __name__ == "__main__":
    test_switchingfunctions()