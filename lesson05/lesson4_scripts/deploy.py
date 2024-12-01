import pickle

# class to deploy model
class DeployClassifier:
    def __init__(self, path):
        self.path = path        

    # function to save model
    def save_model(self, filename, model):
        file_path = self.path + "/" + filename
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

    # function to load model
    def load_model(self, filename):
        file_path = self.path + "/" + filename
        with open(file_path, "rb") as file:
            return pickle.load(file)
