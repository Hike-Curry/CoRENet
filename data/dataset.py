from data.language import add_roles
from data.refsegrs import RefSegRSDataset
from data.rrsisd import ZoomDataset

class GroundedRef(RefSegRSDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_roles(self.samples, kwargs['bert_dir'])

class GroundedRRS(ZoomDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_roles((record for _, records in self.samples for record in records), kwargs['bert_dir'])

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result.pop('native')
        result.pop('native_mask')
        return result
