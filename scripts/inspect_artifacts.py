from pathlib import Path
import torch
import numpy as np
import pandas as pd

def main():
    models_dir = Path('models')
    pths = sorted(models_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pths:
        print('NO_MODEL')
    else:
        p = pths[0]
        ckpt = torch.load(p, map_location='cpu')
        total = 0
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            for v in ckpt['model_state_dict'].values():
                try:
                    total += int(v.numel())
                except Exception:
                    pass
        print('model_path:', p)
        print('param_count:', total)

    hist = Path('results/cnn_smoke_history.csv')
    if hist.exists():
        df = pd.read_csv(hist)
        print('history_epochs:', len(df))
        print('history_last_train_acc', df['train_accuracy'].iloc[-1])
        print('history_last_val_acc', df['val_accuracy'].iloc[-1])
    else:
        print('no history file')

    conf = Path('results/cnn_smoke_confusion.csv')
    if conf.exists():
        cm = np.loadtxt(conf, delimiter=',')
        print('confusion_shape', cm.shape)
        print('confusion_diag', np.diag(cm))
    else:
        print('no confusion file')

if __name__ == '__main__':
    main()
