import os
import sys
import pickle
import torch
import numpy as np

sys.path.insert(0, ".")

from app.models.autoencoder import Autoencoder
from app.models.multi_input import MultiInputClassifier
from app.config import settings

def test():
    model_dir = "insider_threat_model_r52"
    feature_cols = pickle.load(open(os.path.join(model_dir, "feature_cols.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    
    device = torch.device("cpu")
    ae_model = Autoencoder(input_dim=len(feature_cols)).to(device)
    ae_model.load_state_dict(torch.load(os.path.join(model_dir, "autoencoder.pt"), map_location=device, weights_only=True))
    ae_model.eval()

    branches = {
        "intensity": ["logon_count", "file_operations", "email_sent", "email_received",
                      "device_operations", "http_requests", "email_attachments", "email_size_total"],
        "diversity": ["logon_unique_pc", "file_unique_pc", "file_unique_names",
                      "email_unique_recipients", "http_unique_urls"],
        "temporal":  ["after_hours_logons", "after_hours_files", "after_hours_email",
                      "after_hours_device", "after_hours_http", "weekend_logons", "weekend_device"],
        "psychometric": ["O", "C", "E", "A", "N"],
        "anomaly": ["ae_error"],
    }
    branch_dims = {name: len(cols) for name, cols in branches.items()}
    mi_model = MultiInputClassifier(branches=branch_dims, latent_dim=8).to(device)
    mi_model.load_state_dict(torch.load(os.path.join(model_dir, "multi_input.pt"), map_location=device, weights_only=True))
    mi_model.eval()
    
    features = {col: np.random.normal(0, 1) for col in feature_cols}
    raw_vec = np.array([features[c] for c in feature_cols]).reshape(1, -1)
    
    scaled_vec = scaler.transform(raw_vec)
    scaled_tensor = torch.tensor(scaled_vec, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        recon = ae_model(scaled_tensor)
        ae_error = torch.mean((scaled_tensor - recon)**2, dim=1).item()
    features["ae_error"] = ae_error
    
    branch_inputs = {}
    for branch_name, cols in branches.items():
        if branch_name == "anomaly":
            arr = np.array([[ae_error]])
        else:
            branch_vec = []
            for col in cols:
                if col in feature_cols:
                    idx = feature_cols.index(col)
                    branch_vec.append(scaled_vec[0, idx])
                else:
                    branch_vec.append(0.0)
            arr = np.array([branch_vec])
            
        branch_inputs[branch_name] = torch.tensor(arr, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits, attn_weights = mi_model.get_attention(branch_inputs)
        ensemble_score = torch.sigmoid(logits).item()
        
        attn_weights_np = attn_weights.squeeze().cpu().numpy()
        attn_dict = {name: float(attn_weights_np[i]) for i, name in enumerate(mi_model.branch_names)}
        
    print("Score:", ensemble_score)
    print("Attention Weights:", attn_dict)
    print("Test passed successfully")

if __name__ == "__main__":
    test()
