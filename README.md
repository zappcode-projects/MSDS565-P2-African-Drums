# African Drum Classifier: image classification on a custom scraped dataset

MSDS 565, Project 2 (Unit 2, image data). We scraped our own photo dataset of eight traditional African drums, cleaned it, compared three pretrained Keras applications of different sizes with transfer learning, explained what the models look at with saliency maps and Grad-CAM, and deployed all three in a Streamlit app.

**Team:** DataStars &middot; Antonio Brown, Justin Irick, India Scruggs, Michael Flanders

## 1. Classes and why we chose them

| Class | Drum | Origin |
|---|---|---|
| `ashiko` | tapered, conical hand drum | Yoruba, Nigeria |
| `bata` | double-headed, hourglass-shaped drum | Yoruba, Nigeria |
| `bendir` | wide, shallow frame drum | North Africa |
| `bougarabou` | goblet-shaped hand drum, played in sets | Casamance, Senegal |
| `djembe` | goblet drum with a wide head and narrow stem | West Africa |
| `dunun` | cylindrical bass drum | Guinea, West Africa |
| `sabar` | tall, tapered drum played with a stick and the hand | Senegal (Wolof) |
| `talking_drum` | hourglass drum with tension cords | West Africa |

We chose eight classes that can be told apart by **shape** (goblet, hourglass, cylinder, cone, flat frame) and checked beforehand that each could supply at least 200 distinct, correctly labeled photographs. Several are deliberately close pairs (`djembe` and `ashiko`, `dunun` and `talking_drum`), which makes the classification and the explanations informative.

We chose drums because there are many classifications of drums (more than 8 different classifications), and the image types can range from being simple to extremely complex. 

## 2. How the dataset was built

Images were scraped with the course's `ddgs_scraper.py` (the Bing backend), three search terms per class, up to 350 images saved per class:

| Class | Search terms |
|---|---|
| `ashiko` | traditional African ashiko drum &middot; wooden conical ashiko hand drum &middot; African ashiko drum full view |
| `bata` | traditional Yoruba bata drum &middot; Nigerian bata double headed drum &middot; African bata drum isolated |
| `bendir` | North African bendir frame drum &middot; Moroccan bendir traditional drum &middot; Algerian bendir circular drum |
| `bougarabou` | traditional bougarabou African drum &middot; Senegal bougarabou wooden drum &middot; Casamance bougarabou hand drum |
| `djembe` | traditional African djembe drum &middot; West African djembe wooden drum &middot; djembe goblet drum full view |
| `dunun` | traditional African dunun bass drum &middot; West African dundun cylindrical drum &middot; Guinean dunun drum |
| `sabar` | traditional Senegalese sabar drum &middot; Wolof sabar wooden drum &middot; Senegal sabar drum full view |
| `talking_drum` | African talking drum hourglass &middot; Yoruba dundun talking drum &middot; West African talking drum tension cords |

After scraping, the team removed wrong subjects by hand (photos of other instruments, people without a visible drum, collages), then Notebook 2 removed unreadable files and duplicates. 

**No images are in this repository.** The scraped photos, the cleaned images and the preprocessed arrays live in a local folder outside the repo (`~/MSDS565_data/`). `data/P2_manifest.csv` is the record of the dataset: for every image, its class, the search term, the source page URL, and whether it was kept and which split it fell in.

## 3. What each notebook produces, in order

Run the notebooks in order, from the `notebooks/` folder. Each reads what the one before it saved.

| # | Notebook | What it does | Produces |
|---|---|---|---|
| 1 | `P2_Images-1_Scrape.ipynb` | Scrapes the images from the search terms above and records every one | `data/P2_manifest.csv`, the images (local, not committed) |
| 2 | `P2_Images-2_Preprocessing.ipynb` | Cleans, removes duplicates, explores the images, resizes, splits 80/20 (stratified, `random_state=42`) | train/test arrays, label mapping and split table in `~/MSDS565_data/P2_processed/`, and the `kept` and `split` columns in the manifest |
| 3 | `P2_Images-3_TransferLearning.ipynb` | Balances, augments and embeds the training set with three Keras applications; trains a linear probe on each; compares them | `models/P2_N3_<application>.keras` and `models/P2_N3_label_to_index.json` |
| 4 | `P2_Images-4_xAI.ipynb` | Saliency, SmoothGrad and Grad-CAM for the three models; shortcut hunt with ablations | nothing on disk (analysis only) |

Preprocessing decision worth knowing: images are **padded to a square with gray (114, 114, 114)** and resized to 224 &times; 224 (LANCZOS), not stretched or cropped, because the classes are told apart by shape. The Streamlit app copies this exactly.

## 4. Results

Three pretrained Keras applications, each with the same single-layer linear probe and identical training settings (Adam, learning rate 0.001, batch size 200, 500 epochs):

| Application | Size | Parameters | Test accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| MobileNetV2 | small | 2.3 M | 0.76 | 0.755 |
| ResNet50 | medium | 23.6 M | 0.79 | 0.785 |
| EfficientNetV2M | large | 53.2 M | 0.76 | 0.755 |

A larger model did not buy accuracy on this dataset: the medium model was about 3 points better than the small one, and the large model, at 24 times the parameters and more than five times the embedding time, was no better than the small one. Differences of a few points are within the noise of a test set this size.

[TEAM: add two or three sentences from Notebook 4: the weakest classes, whether you found a shortcut and how much the ablation moved accuracy, and what you would fix first.]

## 5. Run the Streamlit app

The app classifies an uploaded photo with all three models, shows each model's sorted probabilities, and overlays raw saliency, SmoothGrad and Grad-CAM maps on the photo.

From the repo root, in the environment described below:

```
streamlit run app/P2_Images-5_App.py
```

It needs `models/` to hold the three `.keras` probes and the label mapping (made by Notebook 3). The first run downloads the pretrained ImageNet weights. The first upload is slow, because SmoothGrad on the large model takes a while; results for the same image are then cached. The visual theme is in `.streamlit/config.toml`, which Streamlit reads when launched from the repo root.

## 6. Environment

```
conda env create -f environment.yml
conda activate msds565
```

`environment.yml` was exported from the environment the notebooks and the app were run in (`conda env export --no-builds`). The scraper needs `ddgs`, and the app needs `streamlit`, both included.

## 7. Repository layout

```
.
├── README.md
├── environment.yml
├── .gitignore
├── .streamlit/
│   └── config.toml               # app theme
├── data/
│   └── P2_manifest.csv           # URLs and labels, no images
├── models/
│   ├── P2_N3_mobilenetv2.keras
│   ├── P2_N3_resnet50.keras
│   ├── P2_N3_effnetv2m.keras
│   └── P2_N3_label_to_index.json
├── notebooks/
│   ├── ddgs_scraper.py           # provided by the course, unmodified
│   ├── image_explainers.py       # provided by the course, unmodified
│   ├── P2_Images-1_Scrape.ipynb
│   ├── P2_Images-2_Preprocessing.ipynb
│   ├── P2_Images-3_TransferLearning.ipynb
│   └── P2_Images-4_xAI.ipynb
└── app/
    ├── image_explainers.py       # provided by the course, a second copy for the app
    └── P2_Images-5_App.py
```

**Model files.** Only the probes are saved, not the pretrained applications, so each `.keras` file is a few hundred kilobytes, far below GitHub's 100 MB limit. **Git LFS is not needed.** The applications are rebuilt with their ImageNet weights at load time, exactly as in Notebook 3.
