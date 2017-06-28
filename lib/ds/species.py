class Species:
    def __init__(self, id, name):
        self.id      = id
        self.name    = name

    def __str__(self):
        return "<Species id={} name={}>".format(self.id, self.name)

    def __repr__(self):
        return self.__str__()
