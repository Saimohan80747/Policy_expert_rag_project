
class Pipeline:

    def __init__(self, steps=[]):
        self.steps = steps

    def run(self, data):
        for step in self.steps:
            data = step(data)
            if(data.get("status","")=="duplicate" or data.get("status","")=="failed" or data.get("status","")=="error"):
                return data
        return data
    
