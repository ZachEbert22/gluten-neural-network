import torch
import pickle
from models.gluten_model import GlutenSubstitutionNet

def load_model_and_vectorizer(model_path="model.pth", vec_path="vectorizer.pkl"):
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    checkpoint = torch.load(model_path, map_location="cpu")
    input_dim = len(vectorizer.get_feature_names_out())
    model = GlutenSubstitutionNet(input_dim=input_dim)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, vectorizer

