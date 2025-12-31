#!/usr/bin/env python3
"""Simple annotation UI for ambiguous marker images.
Run: python scripts/annotation_ui.py --host 0.0.0.0 --port 5000

This lightweight Flask app serves images from data/poc_imgs and lets a reviewer draw rectangles and tag them with a category
('green','amber','red','neutral','ignore'). Submitted JSON is stored under data/annotations/<image_name>.json.
A separate ingestion script can convert annotations into training crops.
"""
from pathlib import Path
import json
import argparse
import os
try:
    from flask import Flask, send_from_directory, request, render_template_string, jsonify
except Exception:
    Flask = None

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / 'data' / 'poc_imgs'
ANN_DIR = ROOT / 'data' / 'annotations'
ANN_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_INDEX = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Annotation UI</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 20px; }
      img { max-width: 100%; }
      .tools { margin-top: 10px; }
      #canvas { border: 1px solid #aaa; touch-action: none; }
    </style>
  </head>
  <body>
    <h2>Annotation UI - Images</h2>
    <ul>
    {% for im in images %}
      <li><a href="/annotate/{{ im }}">{{ im }}</a></li>
    {% endfor %}
    </ul>
  </body>
</html>
"""

TEMPLATE_ANNOTATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Annotate {{ image }}</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 10px; }
      #canvas { border: 1px solid #888; display:block; }
      .controls { margin-top:10px; }
      button { margin-right: 8px; }
    </style>
  </head>
  <body>
    <h3>Annotate: {{ image }}</h3>
    <img id="img" src="/img/{{ image }}" style="display:none;" crossorigin="anonymous"/>
    <canvas id="canvas"></canvas>
    <div class="controls">
      <label>Category:
        <select id="cat">
          <option value="green">green</option>
          <option value="amber">amber</option>
          <option value="red">red</option>
          <option value="neutral">neutral</option>
          <option value="ignore">ignore</option>
        </select>
      </label>
      <button id="save">Save</button>
      <button id="clear">Clear</button>
      <button id="export">Export crops</button>
      <span id="status"></span>
    </div>
    <script>
      const img = document.getElementById('img');
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const status = document.getElementById('status');
      let rects = [];
      let dragging = false; let startX=0, startY=0;
      img.onload = function(){
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
      }
      img.src = '/img/{{ image }}';
      canvas.addEventListener('pointerdown', (e)=>{
        dragging = true; const rect = canvas.getBoundingClientRect();
        startX = (e.clientX - rect.left); startY = (e.clientY - rect.top);
      });
      canvas.addEventListener('pointermove', (e)=>{
        if(!dragging) return;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left); const y = (e.clientY - rect.top);
        // redraw
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.drawImage(img,0,0);
        for(let r of rects){ ctx.strokeStyle = 'lime'; ctx.lineWidth=2; ctx.strokeRect(r.x,r.y,r.w,r.h); ctx.fillStyle='rgba(0,0,0,0.4)'; ctx.fillText(r.cat, r.x+4, r.y+14); }
        ctx.strokeStyle = 'red'; ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, x-startX, y-startY);
      });
      canvas.addEventListener('pointerup', (e)=>{
        dragging=false; const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left); const y = (e.clientY - rect.top);
        const w = x - startX; const h = y - startY;
        rects.push({x: Math.min(startX,x), y: Math.min(startY,y), w: Math.abs(w), h: Math.abs(h), cat: document.getElementById('cat').value});
        // redraw
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.drawImage(img,0,0);
        for(let r of rects){ ctx.strokeStyle = 'lime'; ctx.lineWidth=2; ctx.strokeRect(r.x,r.y,r.w,r.h); ctx.fillStyle='rgba(0,0,0,0.4)'; ctx.fillText(r.cat, r.x+4, r.y+14); }
      });
      document.getElementById('clear').addEventListener('click', ()=>{ rects=[]; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0); status.textContent=''; });
      document.getElementById('save').addEventListener('click', ()=>{
        fetch('/save/{{ image }}', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rects})}).then(r=>r.json()).then(j=>{ status.textContent = j.msg; });
      });
      document.getElementById('export').addEventListener('click', ()=>{
        fetch('/export/{{ image }}', {method:'POST'}).then(r=>r.json()).then(j=>{ status.textContent = j.msg; });
      });
    </script>
  </body>
</html>
"""


if Flask is None:
    print("Flask not available. Install Flask to use the annotation UI: pip install flask")
    exit(0)

app = Flask(__name__)


@app.route('/')
def index():
    images = sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    return render_template_string(TEMPLATE_INDEX, images=images)


@app.route('/img/<path:img_name>')
def serve_img(img_name):
    return send_from_directory(str(IMG_DIR), img_name)


@app.route('/annotate/<path:img_name>')
def annotate(img_name):
    return render_template_string(TEMPLATE_ANNOTATE, image=img_name)


@app.route('/save/<path:img_name>', methods=['POST'])
def save(img_name):
    try:
        data = request.get_json() or {}
        rects = data.get('rects', [])
        outp = {'image': img_name, 'rects': rects}
        p = ANN_DIR / (img_name + '.json')
        p.write_text(json.dumps(outp, indent=2))
        return jsonify({'ok': True, 'msg': f'saved {p}'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})


@app.route('/export/<path:img_name>', methods=['POST'])
def export_crops(img_name):
    # read annotation and make crops for training
    import PIL.Image as PILImage
    p = ANN_DIR / (img_name + '.json')
    if not p.exists():
        return jsonify({'ok': False, 'msg': 'no annotation found'})
    j = json.loads(p.read_text())
    img_p = IMG_DIR / img_name
    im = PILImage.open(img_p)
    out_dir = ROOT / 'data' / 'labels'
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for r in j.get('rects', []):
        x = int(r['x']); y = int(r['y']); w = int(r['w']); h = int(r['h']); cat = r.get('cat','neutral')
        if w <= 2 or h <= 2:
            continue
        crop = im.crop((x, y, x + w, y + h))
        label_dir = out_dir / cat
        label_dir.mkdir(parents=True, exist_ok=True)
        fn = label_dir / f"{img_name}_{count}.png"
        crop.save(fn)
        count += 1
    return jsonify({'ok': True, 'msg': f'exported {count} crops to {out_dir}'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
