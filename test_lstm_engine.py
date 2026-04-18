import torch
from engines.lstm_engine import LSTMEngine
from utils import prepare_data, TranslationDataset, collate_fn
from torch.utils.data import DataLoader
import sys

def run_test():
    import sys
    import io
    # 强制输出为 UTF-8 避免 GBK 报错
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("--- Starting LSTM Engine Standalone Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # 1. Prepare data
    try:
        print("[*] Preparing test data...")
        input_lang, output_lang, pairs = prepare_data(max_length=10, min_freq=1)
        dataset = TranslationDataset(pairs[:100], input_lang, output_lang) 
        loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        print(f"[OK] Data ready: {len(dataset)} samples")
    except Exception as e:
        print(f"[ERROR] Data prep failed: {str(e)}")
        return

    # 2. Init Engine
    try:
        print("[*] Initializing engine...")
        engine = LSTMEngine(device)
        engine.initialize_model(input_lang.n_words, output_lang.n_words, lr=1e-3)
        print("[OK] Engine initialized")
    except Exception as e:
        print(f"[ERROR] Engine init failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test One Epoch
    try:
        print("[*] Starting test training (Epoch 1)...")
        for i, loss in enumerate(engine.train_one_epoch(loader, 1)):
            print(f"    - Batch {i} | Loss: {loss:.4f}")
            if i > 2: 
                break
        print("[OK] Test training success! Data flow passed.")
        
        print("[*] Executing test validation...")
        val_loss = engine.validate(loader)
        print(f"[OK] Test validation success! Loss: {val_loss:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Error during training/validation:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
