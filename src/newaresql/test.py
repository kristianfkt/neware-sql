class Test:
    def __init__(self, test: dict):
        self._test = test

    def __getitem__(self, key):
        return self._test[key]

    def __repr__(self):
        return self._test.__repr__()

    def __str__(self):
        return self._test.__str__()

    def get(self):
        pass

    def stream(self):
        pass

    def plot(self):
        pass
