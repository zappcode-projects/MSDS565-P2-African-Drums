"""
Saliency and Grad-CAM maps for MSDS 565 P2.

Every function here takes the same two things:

  * your joined model -- the Keras application with your trained simple model
    on top, built exactly as in U2-2_CNN-6_TransferLearning section 5:

        model = keras.Sequential([keras.Input((224, 224, 3)), keras_app, probe])

  * one image, exactly as it came out of your Notebook 2 preprocessing: resized,
    RGB, raw 0-255 pixels. Do NOT apply preprocess_input to it first -- these
    functions apply your Keras application's own preprocess_input themselves.

Everything else is handled inside: the preprocess_input, finding the feature maps
Grad-CAM needs, and reading your model's class scores BEFORE its softmax.

    import keras
    from keras.applications import MobileNetV2
    import image_explainers as xai

    # The SAME Keras application call you used in Notebook 3 ...
    keras_app = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                            input_shape=(224, 224, 3))
    # ... the simple model you trained on its embedding vectors there ...
    probe = keras.models.load_model('../models/P2_N3_mobilenetv2.keras')
    # ... joined into one model, as in U2-2_CNN-6_TransferLearning section 5.
    model = keras.Sequential([keras.Input((224, 224, 3)), keras_app, probe])

    xai.explain(model, X_test[0], class_names, true_label=y_test[0])

Or one map at a time. Each is a 2-D array the size of the image, where a bigger
value means "mattered more to the predicted class":

    xai.saliency(model, img)              # one raw gradient
    xai.saliency(model, img, smooth=32)   # SmoothGrad
    xai.grad_cam(model, img)

    xai.predict_proba(model, img)         # the class probabilities, from the raw image

These are the methods from U2-2_CNN-8_Explainability, rewritten for a model with
any number of classes. That notebook's section 3 explains what each one measures.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


# ================================================================
# 1. Each Keras application's own preprocess_input
# ================================================================
# Each Keras application expects its input scaled its own way, and ships its own
# preprocess_input that does the scaling. Keras gives every application model a
# name (MobileNetV2 is 'mobilenetv2_1.00_224', ResNet50 is 'resnet50', ...), so
# the name tells us which preprocess_input to use.
#
# Checked top to bottom, so the more specific names (mobilenetv2, efficientnetv2,
# resnet50v2) have to come before the shorter names they start with.
_PREPROCESSORS = [
    ('mobilenetv2',         keras.applications.mobilenet_v2.preprocess_input),
    ('mobilenetv3',         keras.applications.mobilenet_v3.preprocess_input),
    ('mobilenet',           keras.applications.mobilenet.preprocess_input),
    ('efficientnetv2',      keras.applications.efficientnet_v2.preprocess_input),
    ('efficientnet',        keras.applications.efficientnet.preprocess_input),
    ('resnet50v2',          keras.applications.resnet_v2.preprocess_input),
    ('resnet101v2',         keras.applications.resnet_v2.preprocess_input),
    ('resnet152v2',         keras.applications.resnet_v2.preprocess_input),
    ('resnet',              keras.applications.resnet.preprocess_input),
    ('convnext',            keras.applications.convnext.preprocess_input),
    ('densenet',            keras.applications.densenet.preprocess_input),
    ('xception',            keras.applications.xception.preprocess_input),
    ('inception_resnet_v2', keras.applications.inception_resnet_v2.preprocess_input),
    ('inception_v3',        keras.applications.inception_v3.preprocess_input),
    ('nasnet',              keras.applications.nasnet.preprocess_input),
    ('vgg16',               keras.applications.vgg16.preprocess_input),
    ('vgg19',               keras.applications.vgg19.preprocess_input),
]


def find_preprocess_input(keras_app):
    """Return the preprocess_input function that belongs to a Keras application."""
    name = keras_app.name.lower()
    for prefix, preprocess in _PREPROCESSORS:
        if name.startswith(prefix):
            return preprocess

    raise ValueError(
        f"Could not tell which Keras application '{keras_app.name}' is, so could not "
        f"pick its preprocess_input. Use a model from keras.applications, loaded as in Notebook 3."
    )


# ================================================================
# 2. Feature maps and logits
# ================================================================
# The explainers need two things the joined model does not hand back directly.
#
#   * LOGITS, not probabilities. Your model's last step is softmax, and a
#     confident softmax is flat: once the predicted class sits at 0.999, nudging
#     the image barely moves it, so its gradient is close to zero and the maps
#     come out as noise. A Dense layer computes  x @ W + b  and only then applies
#     its activation, so we compute  x @ W + b  ourselves and stop there.
#
#   * FEATURE MAPS, for Grad-CAM. With pooling='avg', the Keras application
#     averages its last stack of feature maps (7x7 for most applications at
#     224x224) down to one vector. Grad-CAM needs the maps from just before that.
#
# _split_model() finds the pieces once per model; _feature_maps_and_logits()
# runs a batch of images through them.

_parts_cache = {}      # id(model) -> (model, parts); holding the model keeps its id unique


def _split_model(model):
    """Find the Keras application and the simple model on top of it inside a joined model."""
    if id(model) in _parts_cache:
        return _parts_cache[id(model)][1]

    how_to_build = ("Join them as in U2-2_CNN-6_TransferLearning section 5:  "
                    "model = keras.Sequential([keras.Input((224, 224, 3)), keras_app, probe])")

    layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]

    # The Keras application must come first. It is a whole model in its own right,
    # with layers inside it that still output feature maps:
    # (batch, height, width, channels) -- four dimensions.
    keras_app = layers[0] if layers else None
    if not isinstance(keras_app, keras.Model):
        raise ValueError("Expected the Keras application first in your model. " + how_to_build)
    map_layers = [layer for layer in keras_app.layers
                  if not isinstance(layer, keras.layers.InputLayer)
                  and len(layer.output.shape) == 4]
    if not map_layers:
        raise ValueError(f"'{keras_app.name}' is not a Keras application. " + how_to_build)
    if len(keras_app.output.shape) != 2:
        raise ValueError(
            f"'{keras_app.name}' outputs feature maps of shape {tuple(keras_app.output.shape)}, "
            f"not one vector per image. Load it with pooling='avg', exactly as in Notebook 3."
        )

    # A second view of the SAME Keras application (same weights, nothing copied)
    # that returns both its last feature maps and the usual embedding vector.
    last_maps = map_layers[-1]
    maps_and_embedding = keras.Model(keras_app.input, [last_maps.output, keras_app.output])

    # Everything after the Keras application is your simple model -- either one
    # nested model (probe) or its layers listed directly. Collect its layers in
    # order and find the final Dense layer. Anything after that Dense layer (a
    # separate Softmax layer, say) is skipped.
    probe_layers = []
    for layer in layers[1:]:
        if isinstance(layer, keras.Model):
            probe_layers += [sub for sub in layer.layers
                             if not isinstance(sub, keras.layers.InputLayer)]
        else:
            probe_layers.append(layer)
    dense_layers = [layer for layer in probe_layers if isinstance(layer, keras.layers.Dense)]
    if not dense_layers:
        raise ValueError("Could not find a Dense layer after the Keras application. " + how_to_build)
    dense = dense_layers[-1]
    before_dense = probe_layers[:probe_layers.index(dense)]

    parts = {
        'keras_app': keras_app,
        'preprocess': find_preprocess_input(keras_app),
        'maps_and_embedding': maps_and_embedding,
        'before_dense': before_dense,
        'dense': dense,
    }
    _parts_cache[id(model)] = (model, parts)
    return parts


def _feature_maps_and_logits(model, images):
    """Run a batch of raw 0-255 images through the model; return (feature maps, logits)."""
    parts = _split_model(model)

    # A tensor, not a numpy array: preprocess_input changes numpy arrays in
    # place, which would quietly rescale the caller's images.
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    x = parts['preprocess'](images)
    feature_maps, embedding = parts['maps_and_embedding'](x, training=False)

    # Any layers of your simple model before its final Dense layer (Dropout,
    # which does nothing outside training, or hidden Dense layers).
    for layer in parts['before_dense']:
        embedding = layer(embedding, training=False)

    # The final Dense layer WITHOUT its softmax: just  x @ W + b.
    dense = parts['dense']
    logits = tf.matmul(embedding, dense.kernel)
    if dense.use_bias:
        logits = logits + dense.bias

    return feature_maps, logits


def _check_image(model, img):
    """Return img as a float32 array, after checking it is one raw image the model's size."""
    parts = _split_model(model)            # first, check the model is built the right way
    img = np.asarray(img, dtype='float32')

    expected = tuple(parts['keras_app'].input.shape[1:])
    if img.shape != expected:
        raise ValueError(
            f"Expected ONE image of shape {expected} (height, width, RGB), got {img.shape}. "
            f"Resize it with your Notebook 2 strategy first, and pass X[i], not X[i:i+1]."
        )
    if img.min() < 0 or img.max() <= 1.0:
        raise ValueError(
            f"This image's pixels run from {img.min():.2f} to {img.max():.2f}, so it looks "
            f"already scaled. Pass the raw 0-255 image -- these functions apply "
            f"preprocess_input themselves."
        )
    return img


def predict_proba(model, img):
    """Class probabilities for one raw 0-255 image, as a 1-D numpy array."""
    img = _check_image(model, img)
    _, logits = _feature_maps_and_logits(model, img[None])
    return tf.nn.softmax(logits[0]).numpy()


# ================================================================
# 3. The two explainers
# ================================================================
# Every explainer explains ONE class's score. By default that is the class the
# model predicted; pass class_index to ask about a different class instead
# (useful on a misclassified image: "what evidence did it see for the RIGHT class?").

def saliency(model, img, smooth=0, noise=0.10, class_index=None):
    """
    Gradient saliency (Simonyan et al., 2013), with optional SmoothGrad.

    |d logit / d pixel|, maxed over the RGB channels: which pixels would change
    the class score fastest if nudged. One backward pass, pixel-precise, and
    famously speckled.

    smooth > 0 switches on SmoothGrad (Smilkov et al., 2017): average the gradient
    over `smooth` noisy copies of the image. The noise cancels; the structure the
    model responds to consistently survives. `noise` is the noise level as a
    fraction of the 0-255 pixel range.
    """
    img = _check_image(model, img)
    if class_index is None:
        class_index = int(np.argmax(predict_proba(model, img)))

    # The batch of images to differentiate: the image itself, or noisy copies.
    if smooth == 0:
        batch = img[None]
    else:
        batch = img[None] + np.random.normal(0, noise * 255.0, size=(smooth,) + img.shape)
    batch = batch.astype('float32')

    # Gradients in chunks of 8 images, so SmoothGrad on a large Keras
    # application does not run out of memory.
    grads = []
    for start in range(0, len(batch), 8):
        chunk = tf.convert_to_tensor(batch[start:start + 8])
        with tf.GradientTape() as tape:
            tape.watch(chunk)
            _, logits = _feature_maps_and_logits(model, chunk)
            score = logits[:, class_index]
        grads.append(tape.gradient(score, chunk).numpy())
    grads = np.concatenate(grads)                     # (n, height, width, 3)

    heat = np.abs(grads).mean(axis=0)                 # average over the noisy copies
    return heat.max(axis=-1)                          # max over R, G, B


def grad_cam(model, img, class_index=None):
    """
    Grad-CAM (Selvaraju et al., 2017).

    For each channel k of the Keras application's last feature maps A^k, ask how
    much raising that channel would raise the class's logit. That gradient,
    averaged over the map, is the channel's weight; the map is the weighted sum of
    the channels, keeping only the positive part (evidence FOR the class):

        alpha_k = mean over (i, j) of  d logit / d A^k_ij
        map     = ReLU( sum_k alpha_k * A^k )

    Coarse on purpose -- the feature maps are 7x7 for most Keras applications at
    224x224, upsampled to the image size -- so it shows WHERE the evidence is,
    not which pixels.
    """
    img = _check_image(model, img)
    x = tf.convert_to_tensor(img[None])

    # Watching the input records everything computed from it, which is what lets
    # us take the gradient with respect to the feature maps in the middle.
    with tf.GradientTape() as tape:
        tape.watch(x)
        feature_maps, logits = _feature_maps_and_logits(model, x)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        score = logits[:, class_index]

    grads = tape.gradient(score, feature_maps)                      # (1, h, w, channels)
    weights = tf.reduce_mean(grads, axis=(1, 2))                    # (1, channels)
    cam = tf.nn.relu(tf.einsum('bhwc,bc->bhw', feature_maps, weights))

    cam = tf.image.resize(cam[..., None], img.shape[:2], method='bilinear')
    return cam[0, ..., 0].numpy()


# ================================================================
# 4. Displaying the maps
# ================================================================

def norm(heat, pct=99):
    """Scale a map to 0-1, clipping at its 99th percentile so one hot pixel can't wash it out."""
    heat = np.maximum(heat, 0)
    top = np.percentile(heat, pct)
    return np.clip(heat / (top + 1e-8), 0, 1)


def overlay(img, heat, alpha=0.55):
    """
    Blend a map over its image and return the result as a uint8 RGB image.

    Handy in Streamlit, where st.image(overlay(img, heat)) shows it directly.
    """
    colored = matplotlib.colormaps['jet'](norm(heat))[..., :3]     # (h, w, 3), 0-1
    blended = (1 - alpha) * (np.asarray(img) / 255.0) + alpha * colored
    return (blended * 255).astype('uint8')


def explain(model, img, class_names, true_label=None, smooth=32, show=True):
    """
    Plot an image beside all three maps for its predicted class: raw saliency,
    SmoothGrad saliency, and Grad-CAM.

    class_names -- list of class names, in label-index order
    true_label  -- the image's true label index, if you know it (shown in the title)
    show        -- True draws the figure in a notebook; False returns it instead,
                   for st.pyplot(fig) in Streamlit
    """
    img = _check_image(model, img)
    probs = predict_proba(model, img)
    pred = int(np.argmax(probs))

    maps = [
        ('saliency (raw)',        saliency(model, img, class_index=pred)),
        ('saliency (SmoothGrad)', saliency(model, img, smooth=smooth, class_index=pred)),
        ('Grad-CAM',              grad_cam(model, img, class_index=pred)),
    ]

    title = f'pred: {class_names[pred]} ({probs[pred]:.2f})'
    if true_label is not None:
        title = f'true: {class_names[int(true_label)]}   ' + title

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
    axes[0].imshow(img.astype('uint8'))
    axes[0].set_title(title, fontsize=9)
    axes[0].axis('off')
    for ax, (name, heat) in zip(axes[1:], maps):
        ax.imshow(img.astype('uint8'))
        ax.imshow(norm(heat), cmap='jet', alpha=0.55)
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig
