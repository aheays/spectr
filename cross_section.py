from .dataset import Dataset

class CrossSection(Dataset):
    """Common stuff for for lines and levels."""
    
    def __init__(self,name=None,**kwargs):
        Dataset.__init__(
            self,
            name=name,
            prototypes=dict(
                ν = dict(description="Frequency",kind='f',units='cm-1'),
                σ = dict(description="Cross section",kind='f',units='cm2'),
            ),
            **kwargs)
        # self._cache = {}
        # self.pop_format_input_function()
        # self.automatic_format_input_function(limit_to_args=('name',))
        # self.default_zkeys = self._defining_qn
