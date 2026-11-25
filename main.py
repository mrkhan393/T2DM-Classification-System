from config import features
import os, warnings, torch
from utils.device import select_torch_device, synchronize_device
from models.model import LSTMContrastiveClassifier
from models.losses import WeightedFocalClassificationLoss
from data_pipeline.data_preparation import load_data
from federated.federated_setup import run_federated
from evaluation.final_evaluation import run_final_evaluation
from optimization.model_compression import run_compression
from utils.audit import audit_outputs

def verify_training_backend(candidate):
    probe = LSTMContrastiveClassifier(input_dim=1).to(candidate)
    probe.train()
    xb = torch.randn(6, len(features), 1, device=candidate, dtype=torch.float32)
    yb = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], device=candidate)
    loss_fn = WeightedFocalClassificationLoss(2.0, 1.0).to(candidate)
    opt = torch.optim.Adam(probe.parameters(), lr=0.0001)
    embs, logits = probe(xb, return_embedding=True)
    loss = loss_fn(logits, yb) + 0.5 * (embs @ embs.T).mean()
    loss.backward()
    opt.step()
    synchronize_device(candidate)
    assert torch.isfinite(loss).item(), 'Non-finite probe loss'
    del probe, opt, xb, yb, embs, logits, loss

def main():
    device = select_torch_device()
    data = load_data()
    try:
        verify_training_backend(device)
        print('Training backend forward/backward probe: passed on', device)
    except Exception as exc:
        if device.type != 'mps' or os.environ.get('DIAFED_DEVICE', 'auto') != 'auto':
            raise
        warnings.warn(f'MPS LSTM training probe failed ({type(exc).__name__}: {exc}); falling back to CPU.')
        device = torch.device('cpu')
        verify_training_backend(device)
        print('Training backend forward/backward probe: passed on CPU')
    result = run_federated(data, device)
    run_final_evaluation(data, result)
    run_compression(data, result, device)
    audit_outputs(data, result)
if __name__ == '__main__':
    main()
