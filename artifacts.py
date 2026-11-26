import contextlib
import json
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from utils import atomic_save
VERSION = 'ccsnet'
POLICY = 'ccsnet-validation-best'

@contextlib.contextmanager
def preserve_rng():
    py, np_state = (random.getstate(), np.random.get_state())
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)

def publish_best(stage, output):
    output, stage = (Path(output).resolve(), Path(stage).resolve())
    if stage.parent != output or not stage.name.startswith('.grounded_pending_'):
        raise ValueError('Unmanaged stage path')
    destination = output / 'best'
    if destination.is_symlink():
        raise ValueError('Refusing symlink destination')
    if destination.exists():
        marker = destination / 'best_summary.json'
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf8')).get('artifact_policy') != POLICY:
            raise FileExistsError('Existing best is not managed by this method; use a new output')
    with tempfile.TemporaryDirectory(prefix='.grounded_previous_', dir=output) as temporary:
        previous = Path(temporary).resolve()
        if previous.parent != output:
            raise ValueError('Unsafe backup path')
        backup = previous / 'best'
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise

def save_best(output, model, optimizer, scheduler, scaler, args, epoch, best, steps, record):
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix='.grounded_pending_', dir=output) as temporary:
        stage = Path(temporary)
        summary = dict(version=VERSION, artifact_policy=POLICY, epoch=epoch, selected_by='val.mIoU', val=record['val'], checkpoint='best.pth')
        (stage / 'best_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
        payload = dict(version=VERSION, config=vars(args), epoch=epoch, best_mIoU=best, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), steps_per_epoch=steps, selected_by=args.select_best_by + '.mIoU', rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(stage / 'best.pth', payload)
        publish_best(stage, output)
