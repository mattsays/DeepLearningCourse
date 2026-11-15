# Lie Detection on DOLOS: Experimental Protocol and Architectures
## Part 2: The 3 Architectural Paradigms

Training from scratch, or fully fine-tuning large audio-video models on DOLOS, leads to fast overfitting
and saturates VRAM. The three paradigms below avoid the problem by exploiting *pre-trained* models through
three distinct principles from the recent literature.

---

### Paradigm 1: Frozen Foundation Models + Cross-Modal Fusion (PECL-inspired) --TIA

#### Theory
This paradigm takes its cue directly from **PECL (Parameter-Efficient Crossmodal Learning)**, the
architecture presented in the original paper for the DOLOS dataset (ICCV 2023).
The underlying idea is that modern foundation models — pre-trained self-supervised on millions of images
and thousands of hours of speech — already hold a near-perfect latent representation of the human face and
of the paralinguistic nuances of the voice. Trying to move the weights of these giants on a dataset as
small as DOLOS would only risk "forgetting" that prior knowledge (*catastrophic forgetting*).

The principle is therefore to **freeze the unimodal extractors entirely (100%)** and concentrate all the
learning capacity on the **multimodal fusion mechanism**. Deception, in fact, often emerges from the
*incongruence* between channels (a steady voice paired with a micro-expression of tension, or the other way
round). Using *Cross-Attention*, the model learns to query the visual stream with the acoustic stream as
*Query* (and vice versa), aligning the two signals in time at negligible computational cost.

*Ideas for variations on the theme:*
* Try different fusion strategies: symmetric *Co-Attention*, *Bottleneck Attention* (a few latent tokens
  mediating between audio and video, *Perceiver*-style), or *Gated Multimodal Units* (GMU).
* Extract features from the penultimate layers of the backbones rather than the last one: intermediate
  layers often retain finer, less abstract spatial/acoustic information.

#### Implementation
* **Video backbone (frozen):** **DINOv2** or **SigLIP**. Unlike classic ViTs trained to classify generic
  objects, DINOv2 captures very fine geometric and local detail — ideal for micro-expressions. Pass the 16
  frames one by one through the model under `torch.no_grad()` to get a sequence of 16 embedding vectors.
  * *Download:* [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) or
    [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) (via
    `transformers`).
* **Audio backbone (frozen):** **WavLM**. It is Microsoft's evolution of Wav2Vec 2.0; during pre-training it
  was also trained on speech separation and paralinguistic tasks, which makes it better at capturing vocal
  stress, pitch and hesitation.
  * *Download:* [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus) (via
    `transformers`).
* **Trainable module (plain PyTorch):**
  1. Two linear layers (`nn.Linear`) to project the DINOv2 ($384$ dim) and WavLM ($768$ dim) embeddings into
     the same common dimension (e.g. $d_{\text{model}} = 256$).
  2. An `nn.MultiheadAttention` module to compute Cross-Attention between the video and audio temporal
     sequences, followed by an `nn.TransformerEncoderLayer` and a final MLP classifier.
  3. **Time-saving trick:** because the backbones are frozen, you can run a one-off pre-extraction pass and
     save the DINOv2 and WavLM output tensors to disk (`.pt`). Training the fusion network then takes less
     than a minute per epoch.

---

### Paradigm 2: Spatio-Temporal Transformers with PEFT / LoRA Fine-Tuning --LORE

#### Theory
Unlike Paradigm 1 (which analyses frames as separate still images and then merges them), this approach uses
**native spatio-temporal Transformers**. Models such as VideoMAE or ViViT do not look at a single frame:
they split the whole clip into 3D "tubes" (*3D tubelets*) and apply attention over space and time
simultaneously from the very first layer.

However, because these models were pre-trained on macroscopic action datasets (such as *Kinetics-400*:
people playing football, swimming or dancing), their attention matrices are not immediately tuned to notice
an imperceptible blink or a lip contraction. It is therefore necessary to adapt the internal weights of the
backbone without causing overfitting or running out of GPU memory.

The theoretical answer is **PEFT (Parameter-Efficient Fine-Tuning)**, specifically through **LoRA (Low-Rank
Adaptation)**. LoRA assumes that the weight change needed to adapt a large model to a new task has a "low
intrinsic rank". Instead of updating a weight matrix $W \in \mathbb{R}^{d \times k}$, it freezes it and adds
two tiny matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ (with rank
$r \ll d, k$). That way fewer than 1% of the total parameters are updated, changing the internal behaviour
of the Transformer at minimal VRAM cost.

*Ideas for variations on the theme:*
* Compare PEFT methods other than LoRA, such as **DoRA** (Weight-Decomposed Low-Rank Adaptation) or
  **AdaLoRA**, both already available in the same library.
* Test where the adapters go: apply LoRA only to the attention layers (`query`, `value`), or include the
  feed-forward (MLP) layers of the last Transformer blocks as well.
* Explore different *Late Fusion* techniques (weighted average of the logits, concatenation of the hidden
  states before the classifier, or *confidence-weighted fusion*).

#### Implementation
* **Video model (with LoRA):** **VideoMAE** (Video Masked Autoencoder). It is currently the most solid
  single-GPU standard for video understanding, far more memory-efficient than the early ViViT. It takes an
  input tensor of shape `(Batch, 16, 3, 224, 224)`.
  * *Download:* [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
    or the V2 version [`OpenGVLab/VideoMAEv2-Base`](https://huggingface.co/OpenGVLab/VideoMAEv2-Base).
* **Audio model (with LoRA):** **HuBERT**, or just the acoustic encoder of **Whisper**. HuBERT learns
  discrete hidden acoustic units and is a reference point in the literature on emotion and deception
  analysis.
  * *Download:* [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960) or
    [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (using `WhisperModel.encoder`).
* **Training library:** **HuggingFace PEFT** (`pip install peft`).
  * Wrap both models with
    `get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"]))`.
  * To stay within Colab/laptop memory: use `batch_size = 2` or `4` together with
    `gradient_accumulation_steps = 8` and mixed-precision training (`fp16` or `bf16`).

---

### Paradigm 3: Modern 3D Convolutional Architectures + Sequential Modelling --FAB

#### Theory
Vision Transformers (Paradigms 1 and 2) lack an *inductive bias*: they do not know what "closeness" between
pixels, or between contiguous instants in time, means until they learn it from enormous amounts of data. On
small, noisy datasets like DOLOS, **Convolutional Networks (CNNs)** often provide better natural
regularisation thanks to the locality of convolutional filters and translation invariance.

This paradigm evolves the classic *I3D + LSTM* architecture by using modern, extremely light spatio-temporal
convolutions to extract local motion patterns over short temporal windows, paired with a time-frequency
representation (spectrogram) for the audio. Audio is not treated as a 1D wave but converted into a **2D Mel
Spectrogram** — an image whose X axis is time, Y axis is the frequency perceived by the human ear, and
colour is intensity. That makes it possible to use the latest computer-vision architectures on speech too,
capturing harmonics, tremors in the fundamental frequency $F_0$ and silences.

Finally, the long-range dynamics of the whole clip are delegated to a dedicated sequential classifier (a
bidirectional recurrent network, or a dilated temporal convolutional network) that reads how the features
evolve from the beginning to the end of the subject's answer.

*Ideas for variations on the theme:*
* Replace the classic RNNs (LSTM/GRU) in the final classifier with a **TCN (Temporal Convolutional
  Network)** or a **Mamba** block (State Space Model): both handle long sequences better without suffering
  from *vanishing gradients*.
* Split the video into short overlapping *chunks* (e.g. 8-frame windows with stride 4), extract one vector
  per window with the 3D CNN, and feed the sequence of windows to the recurrent module.

#### Implementation
* **Video model (3D / hybrid CNN):** Meta's **X3D** (Expand 3D) or **Swin3D**. X3D is a 3D convolutional
  network designed by progressively expanding the axes: it outperforms I3D and SlowFast while requiring
  roughly 5× less compute (FLOPs) and very little memory.
  * *Download:* available natively in PyTorch without heavy external dependencies via
    [`torchvision.models.video`](https://pytorch.org/vision/stable/models.html#video-classification)
    (e.g. `swin3d_t(weights='DEFAULT')` or `mvit_v2_s`), or via **PyTorchVideo**
    (`torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)`).
* **Audio model (spectrogram + 2D CNN):**
  1. Turn the `.wav` file into a Mel spectrogram with
     `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)`.
  2. Feed the spectrogram (as a 1- or 3-channel image) into **ConvNeXt-Tiny**, a pure CNN modernised with
     Transformer design principles: light and very strong.
  * *Download:* [`timm.create_model('convnext_tiny.fb_in22k', pretrained=True)`](https://huggingface.co/timm/convnext_tiny.fb_in22k)
    via the `timm` library (`pip install timm`).
* **Trainable sequential module:**
  * Extract the temporal feature sequences from X3D and ConvNeXt, concatenate them along the feature axis
    and feed them to a **Bi-GRU** (`nn.GRU(input_size, hidden_size=128, bidirectional=True,
    batch_first=True)`) with a high *dropout* ($p=0.4$ or $0.5$) to fight overfitting, followed by a linear
    layer for the binary prediction (Truth / Lie).

---

## Part 3: Comparative Summary for the Paper

This structure lets you present a complete comparative study in the final report, covering the three great
families of multimodal deep learning:

| Paradigm | Core idea | Video backbone | Audio backbone | Trained module |
| :--- | :--- | :--- | :--- | :--- |
| **1. PECL-Style** | Frozen static extraction + deep fusion | `dinov2-small` / `siglip` | `wavlm-base-plus` | Cross-Modal Attention Transformer |
| **2. End-to-End PEFT** | Low-rank internal adaptation (LoRA) | `videomae-base` | `hubert-base` / `whisper` | LoRA matrices ($r=8$) + Late Fusion |
| **3. 3D-CNN + Spec** | Convolutional inductive bias + recurrence | `x3d_m` / `swin3d_t` | Mel-Spec + `convnext_tiny` | Bi-GRU / TCN (Temporal ConvNet) |
