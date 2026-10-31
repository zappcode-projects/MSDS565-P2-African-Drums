"""
P2 App: classify an uploaded photo as one of our eight African drum types with
every one of our three models, and show what each model looked at.

Run from the repo root:

    streamlit run app/P2_Images-5_App.py

Pipeline, exactly as in the notebooks:
  Notebook 2  convert to RGB, pad to a square with gray (114, 114, 114), resize to 224 x 224 (LANCZOS)
  Notebook 3  each Keras application (include_top=False, pooling='avg') + its trained probe from models/
  Notebook 4  the same joined model goes to image_explainers, which applies each application's own
              preprocess_input itself, so it is NOT applied again here
"""
import hashlib
import io
import json
import time
from pathlib import Path

import altair as alt
import keras
import numpy as np
import pandas as pd
import streamlit as st
from packaging.version import Version
from PIL import Image, UnidentifiedImageError

import image_explainers as xai          # the provided module, copied beside this script and not modified

# ---------------------------------------------------------------- settings (must match the notebooks)
IMG_SIZE = 224                          # Notebook 2: target size
PAD_COLOR = (114, 114, 114)             # Notebook 2: constant padding color
WEIGHTS = 'imagenet'                    # Notebook 3: the pretrained weights every application is loaded with
SMOOTH = 32                             # Notebook 4: noisy copies averaged by SmoothGrad
SEED = 42

# Streamlit renamed use_container_width to width='stretch' (version 1.50); pick whichever this install understands
FIT = {'width': 'stretch'} if Version(st.__version__) >= Version('1.50') else {'use_container_width': True}

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'      # the repo's models/ folder, wherever it is launched from

APPLICATIONS = {                        # Notebook 3: same applications, same calls, same order (small, medium, large)
    'mobilenetv2': dict(name='MobileNetV2', tier='small',
        build=lambda: keras.applications.MobileNetV2(weights=WEIGHTS, include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))),
    'resnet50': dict(name='ResNet50', tier='medium',
        build=lambda: keras.applications.ResNet50(weights=WEIGHTS, include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))),
    'effnetv2m': dict(name='EfficientNetV2M', tier='large',
        build=lambda: keras.applications.EfficientNetV2M(weights=WEIGHTS, include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))),
}


# ---------------------------------------------------------------- Notebook 2's preprocessing, copied exactly
def resize_pad(img, size=IMG_SIZE, color=PAD_COLOR):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new('RGB', (size, size), color)
    canvas.paste(img.resize((new_w, new_h), Image.Resampling.LANCZOS), ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def preprocess_upload(file_bytes):
    """Uploaded bytes -> (224, 224, 3) uint8 array, raw 0-255, exactly as Notebook 2 saved the test images."""
    with Image.open(io.BytesIO(file_bytes)) as img:
        return np.asarray(resize_pad(img.convert('RGB')), dtype='uint8')


# ---------------------------------------------------------------- models (loaded once per server run)
@st.cache_resource(show_spinner='Loading the three models (the first run downloads the pretrained weights)...')
def load_models():
    label_to_index = json.loads((MODELS_DIR / 'P2_N3_label_to_index.json').read_text())
    class_names = [name for name, i in sorted(label_to_index.items(), key=lambda kv: kv[1])]

    models = {}
    for key, cfg in APPLICATIONS.items():
        keras_app = cfg['build']()
        keras_app.trainable = False
        probe = keras.models.load_model(MODELS_DIR / f'P2_N3_{key}.keras')
        assert probe.output_shape[-1] == len(class_names), f'{key}: probe and label mapping disagree'
        models[key] = dict(model=keras.Sequential([keras.Input((IMG_SIZE, IMG_SIZE, 3)), keras_app, probe]),
                           name=cfg['name'], tier=cfg['tier'], params=keras_app.count_params())
    return class_names, models


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(key, file_bytes):
    """Probabilities and the three maps for one model on one upload. Cached on the file's bytes."""
    _, models = load_models()
    start = time.perf_counter()
    model, img = models[key]['model'], preprocess_upload(file_bytes)
    probs = xai.predict_proba(model, img)
    pred = int(probs.argmax())
    np.random.seed(SEED + int(hashlib.md5(file_bytes).hexdigest(), 16) % 10_000)      # SmoothGrad's noise: repeatable per image
    return dict(probs=probs, pred=pred,
                raw=xai.saliency(model, img, class_index=pred),
                smooth=xai.saliency(model, img, smooth=SMOOTH, class_index=pred),
                cam=xai.grad_cam(model, img, class_index=pred), seconds=time.perf_counter() - start)


def uncertainty(probs):
    """Top probability, its margin over the runner-up, and the entropy as a share of the maximum possible."""
    top2 = np.sort(probs)[::-1][:2]
    entropy = max(0.0, float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs))))      # max() stops a tiny negative from showing as -0.00
    return float(top2[0]), float(top2[0] - top2[1]), entropy


# ---------------------------------------------------------------- look and feel
ACCENT, GREEN, AMBER, RED = '#E8A33D', '#3FB68B', '#E8A33D', '#E5675B'

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, .stMarkdown *, h1, h2, h3, h4, label, p, li, button p, [data-testid="stCaptionContainer"] *,
[data-testid="stFileUploaderDropzoneInstructions"] * { font-family: 'DM Sans', sans-serif; }
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }
.block-container { max-width: 1280px; padding-top: 4rem; padding-bottom: 3rem; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }

.hero { padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(232,163,61,.20), rgba(232,163,61,.04) 55%, rgba(128,128,128,.05));
        border: 1px solid rgba(232,163,61,.35); }
.hero .eyebrow { letter-spacing: .14em; text-transform: uppercase; font-size: .72rem; font-weight: 600; color: #E8A33D; }
.hero h1 { margin: .15rem 0 .35rem 0; padding: 0; font-size: 2.5rem; font-weight: 700; letter-spacing: -.02em; }
.hero p { margin: 0; opacity: .82; font-size: 1.02rem; max-width: 62rem; line-height: 1.55; }

.steps { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1.1rem; }
.step { display: flex; align-items: center; gap: .55rem; padding: .45rem .9rem .45rem .5rem; border-radius: 999px;
        background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.22); font-size: .86rem; font-weight: 500; }
.step b { display: inline-flex; width: 1.5rem; height: 1.5rem; border-radius: 50%; align-items: center; justify-content: center;
          background: #E8A33D; color: #1b1405; font-size: .78rem; }

.card { padding: 1.05rem 1.2rem; border-radius: 14px; background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.22); height: 100%; }
.card h4 { margin: 0 0 .35rem 0; font-size: 1rem; font-weight: 600; }
.card p { margin: 0; opacity: .78; font-size: .92rem; line-height: 1.5; }

.pill { display: inline-block; white-space: nowrap; padding: .12rem .62rem; border-radius: 999px; font-size: .74rem; font-weight: 600; letter-spacing: .02em;
        border: 1px solid; vertical-align: middle; }
.note-box { padding: .85rem 1rem; border-radius: 12px; font-size: .88rem; line-height: 1.55; background: rgba(232,163,61,.09); border: 1px solid rgba(232,163,61,.3); }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .4rem; }
.chip { padding: .18rem .65rem; border-radius: 999px; font-size: .8rem; background: rgba(232,163,61,.14); border: 1px solid rgba(232,163,61,.35); }

.verdict { padding: 1.2rem 1.5rem; border-radius: 16px; margin: .4rem 0 1rem 0; border: 1px solid; }
.verdict .label { font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 600; opacity: .85; }
.verdict .answer { font-size: 2.1rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.2; margin: .1rem 0; }
.verdict .note { opacity: .82; font-size: .95rem; }

.tile .name { font-weight: 600; font-size: .98rem; margin-bottom: .35rem; }
.tile .pred { font-size: 1.45rem; font-weight: 700; letter-spacing: -.01em; }
.tile .prob { font-size: .9rem; opacity: .85; margin: .1rem 0 .55rem 0; }

.mhead { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; margin-bottom: .15rem; }
.mhead .mname { font-size: 1.35rem; font-weight: 700; letter-spacing: -.01em; }
.mhead .meta { opacity: .6; font-size: .86rem; }
.mpred { font-size: 1rem; margin-bottom: .7rem; opacity: .92; }
.sub { font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .65; margin: .15rem 0 .35rem 0; }

.legend { display: flex; align-items: center; gap: .6rem; font-size: .8rem; opacity: .85; margin: .2rem 0 .8rem 0; }
.legend .bar { flex: 0 0 220px; height: 10px; border-radius: 6px;
               background: linear-gradient(90deg,#00007f,#0000ff,#007fff,#00ffff,#7fff7f,#ffff00,#ff7f00,#ff0000,#7f0000); }

[data-testid="stImage"] img { border-radius: 12px; }
[data-testid="stFileUploader"] section { border-radius: 14px; border: 1.5px dashed rgba(232,163,61,.55); background: rgba(232,163,61,.05); }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.foot { opacity: .55; font-size: .8rem; line-height: 1.6; margin-top: 1.6rem; }
</style>
"""


def pill(text, color):
    return f'<span class="pill" style="color:{color};border-color:{color};background:{color}1f">{text}</span>'


def confidence(top):
    """A plain-language reading of the top probability. A rule of thumb, not a calibrated measure."""
    if top >= 0.80:
        return 'High confidence', GREEN
    if top >= 0.50:
        return 'Moderate confidence', AMBER
    return 'Low confidence', RED


def probability_chart(probs, class_names):
    """Horizontal bars of all class probabilities, sorted highest first, the top class in the accent color."""
    df = pd.DataFrame({'class': class_names, 'probability': probs}).sort_values('probability', ascending=False).reset_index(drop=True)
    df['top'] = df.index == 0
    order = df['class'].tolist()
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=17).encode(
        y=alt.Y('class:N', sort=order, title=None, axis=alt.Axis(labelFontSize=12, ticks=False, domain=False)),
        x=alt.X('probability:Q', scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format='%', tickCount=5, grid=True)),
        color=alt.condition(alt.datum.top, alt.value(ACCENT), alt.value('#6f7480')),
        tooltip=[alt.Tooltip('class:N', title='class'), alt.Tooltip('probability:Q', format='.1%')])
    labels = alt.Chart(df).mark_text(align='left', dx=5, fontSize=11, color='#9aa0a6').encode(
        y=alt.Y('class:N', sort=order), x='probability:Q', text=alt.Text('probability:Q', format='.0%'))
    return (bars + labels).properties(height=250)


def show_empty_state(class_names):
    st.markdown('<div class="sub">What you will get</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown('<div class="card"><h4>One answer from three models</h4><p>A small, a medium and a large model classify your photo, '
               'one per row, so you can see whether size changes the answer.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>Every probability, sorted</h4><p>All eight classes with their probabilities, highest first, '
               'so you can see how sure each model is and what it nearly said instead.</p></div>', unsafe_allow_html=True)
    c.markdown('<div class="card"><h4>What each model looked at</h4><p>Raw saliency, SmoothGrad and Grad-CAM maps lay the model\'s '
               'evidence over your photo, so you can check it looked at the drum.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="foot">Works best with a single drum, photographed in full. Try a photo that is not one of our drums as well: '
               'it still gets an answer, which is the point of the warning in the sidebar.</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='African Drum Classifier', page_icon='🥁', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        class_names, models = load_models()
    except Exception as error:                            # a missing model file is the usual cause
        st.error(f'Could not load the models from {MODELS_DIR}: {error}. Run Notebook 3 first, so models/ holds the three probes and the label mapping.')
        st.stop()

    with st.sidebar:
        st.markdown('### 🥁 The eight drums')
        st.markdown('<div class="chips">' + ''.join(f'<span class="chip">{n.replace("_", " ")}</span>' for n in class_names) + '</div>', unsafe_allow_html=True)
        st.markdown('### The models')
        for info in models.values():
            st.markdown(f'**{info["name"]}** &nbsp; {pill(info["tier"], "#9aa0a6")}<br><span style="opacity:.6;font-size:.82rem">{info["params"] / 1e6:.1f} M parameters</span>',
                        unsafe_allow_html=True)
        st.markdown('### What this app cannot tell you')
        st.markdown('<div class="note-box">The models can only answer with one of these eight drums. A photo of anything else still gets a probability '
                    'for each, because the probabilities always add up to 100%. A confident answer is not proof the photo is one of our drums.</div>',
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><div class="eyebrow">Image classification &middot; MSDS 565</div>'
        '<h1>African Drum Classifier</h1>'
        '<p>Upload a photo of a drum. Three models of different sizes each name it, then show <b>what they looked at</b> '
        'with saliency maps and Grad-CAM, so you can judge the answer and not only read it.</p>'
        '<div class="steps"><span class="step"><b>1</b>Upload a photo</span><span class="step"><b>2</b>Padded to a square, 224 &times; 224</span>'
        '<span class="step"><b>3</b>Three models classify it</span><span class="step"><b>4</b>See the evidence</span></div></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader('Upload a drum photo', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'], label_visibility='collapsed')
    if uploaded is None:
        show_empty_state(class_names)
        st.stop()

    file_bytes = uploaded.getvalue()
    try:
        img = preprocess_upload(file_bytes)
    except (UnidentifiedImageError, OSError):
        st.error('That file could not be read as an image. Try a JPG or PNG.')
        st.stop()

    left, right, rest = st.columns([1, 1, 1.5])
    left.image(file_bytes, caption='Your upload', **FIT)
    right.image(img, caption=f'What the models see ({IMG_SIZE} x {IMG_SIZE}, padded)', **FIT)

    # one row per model, so the size comparison is visible at a glance
    results = {}
    progress = st.progress(0.0, text='Running the models...')
    for n, key in enumerate(models):
        progress.progress(n / len(models), text=f'Running {models[key]["name"]} ({n + 1} of {len(models)}); the large model is the slowest...')
        results[key] = analyse(key, file_bytes)
    progress.empty()

    preds = {key: class_names[r['pred']] for key, r in results.items()}
    tops = {key: uncertainty(r['probs']) for key, r in results.items()}
    agree = len(set(preds.values())) == 1
    least_sure = min(t[0] for t in tops.values())

    with rest:                                              # the verdict, beside the images
        if agree:
            color, label, answer = GREEN, 'All three models agree', preds[next(iter(preds))].replace('_', ' ')
            note = f'Lowest confidence among them: {least_sure:.0%}.'
        else:
            color, label, answer = AMBER, 'The models disagree', ' · '.join(dict.fromkeys(p.replace('_', ' ') for p in preds.values()))
            note = 'Models of different sizes giving different answers is a sign to treat every answer with caution.'
        if least_sure < 0.5:
            note += ' At least one model is below 50% on its top class, so it is not committing to an answer (a rule of thumb, not a calibrated threshold).'
        st.markdown(f'<div class="verdict" style="border-color:{color};background:{color}14">'
                    f'<div class="label" style="color:{color}">{label}</div><div class="answer">{answer}</div><div class="note">{note}</div></div>',
                    unsafe_allow_html=True)
    for col, (key, r) in zip(st.columns(len(results)), results.items()):
        top, margin, entropy = tops[key]
        text, tone = confidence(top)
        col.markdown(f'<div class="card tile"><div class="name">{models[key]["name"]} <span style="opacity:.55;font-weight:400">&middot; {models[key]["tier"]}</span></div>'
                     f'<div class="pred">{preds[key].replace("_", " ")}</div>'
                     f'<div class="prob">{top:.0%} probability &middot; {margin * 100:.0f} pts ahead of the runner-up</div>'
                     f'{pill(text, tone)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub" style="margin-top:1.6rem">What each model looked at</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend"><span>less influence</span><span class="bar"></span><span>more influence on the predicted class</span></div>', unsafe_allow_html=True)

    for key, r in results.items():
        info, (top, margin, entropy) = models[key], tops[key]
        text, tone = confidence(top)
        with st.container(border=True):
            st.markdown(f'<div class="mhead"><span class="mname">{info["name"]}</span>{pill(info["tier"], "#9aa0a6")}'
                        f'<span class="meta">{info["params"] / 1e6:.1f} M parameters &middot; maps took {r["seconds"]:.0f} s</span></div>'
                        f'<div class="mpred">Predicts <b>{preds[key].replace("_", " ")}</b> at {top:.0%} &nbsp; {pill(text, tone)}</div>', unsafe_allow_html=True)
            cols = st.columns([1.35, 1, 1, 1])
            cols[0].markdown('<div class="sub">Class probabilities</div>', unsafe_allow_html=True)
            cols[0].altair_chart(probability_chart(r['probs'], class_names), **FIT)
            for col, title, heat in zip(cols[1:], ['Saliency (raw)', 'SmoothGrad', 'Grad-CAM'], [r['raw'], r['smooth'], r['cam']]):
                col.markdown(f'<div class="sub">{title}</div>', unsafe_allow_html=True)
                col.image(xai.overlay(img, heat), **FIT)

    with st.expander('How to read these maps'):
        st.markdown('- **Saliency (raw)** marks the pixels that would change the model\'s score fastest if nudged. It is pixel-precise but speckled.\n'
                    f'- **SmoothGrad** averages that over {SMOOTH} noisy copies of the image, which cancels the speckle and keeps the shape the model responds to.\n'
                    '- **Grad-CAM** shows which regions of the last feature maps support the class. It is coarse (a 7 x 7 grid, stretched) but shows *where* the evidence sits.\n'
                    '- All three explain the **predicted** class. A map suggests what a model uses; it does not prove it.')

    st.markdown('<div class="foot">Preprocessing matches the training pipeline exactly: RGB, padded to a square with gray (114, 114, 114), resized to 224 &times; 224. '
                'Each model is a pretrained Keras application with a small linear classifier trained on top.</div>', unsafe_allow_html=True)


main()
