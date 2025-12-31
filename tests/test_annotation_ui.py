import pytest
pytest.importorskip('flask')
from scripts import annotation_ui
from pathlib import Path
import json

ANN_DIR = annotation_ui.ANN_DIR
IMG_DIR = annotation_ui.IMG_DIR


def test_save_annotation(tmp_path, monkeypatch):
    # create a dummy image file in IMG_DIR
    img_name = 'test_ui_dummy.png'
    p = IMG_DIR / img_name
    # create small png
    from PIL import Image
    img = Image.new('RGB', (50,50), (255,255,255))
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    client = annotation_ui.app.test_client()
    payload = {'rects':[{'x':10,'y':10,'w':20,'h':20,'cat':'green'}]}
    rv = client.post(f'/save/{img_name}', json=payload)
    assert rv.status_code == 200
    j = rv.get_json()
    assert j.get('ok') is True
    ann_file = ANN_DIR / (img_name + '.json')
    assert ann_file.exists()
    data = json.loads(ann_file.read_text())
    assert data.get('image') == img_name
    assert len(data.get('rects', [])) == 1
    # cleanup
    ann_file.unlink()
    p.unlink()