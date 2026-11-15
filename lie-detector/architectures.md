# Progetto Lie Detection su Dataset DOLOS: Protocollo Sperimentale e Architetture
## Parte 2: I 3 Paradigmi Architetturali

L'addestramento da zero (*from scratch*) o il *full fine-tuning* di grandi modelli audio-video su DOLOS porta a un rapido overfitting e satura la VRAM. I tre paradigmi seguenti aggirano il problema sfruttando modelli *pre-trained* tramite tre principi distinti della letteratura recente.

---

### Paradigma 1: Frozen Foundation Models + Cross-Modal Fusion (Ispirato a PECL) --TIA

#### Spiegazione Teorica
Questo paradigma si ispira direttamente all'architettura **PECL (Parameter-Efficient Crossmodal Learning)** presentata nel paper originale del dataset DOLOS (ICCV 2023). 
L'idea teorica di fondo è che i grandi modelli fondazionali moderni (addestrati in modo *self-supervised* su milioni di immagini e migliaia di ore di parlato) possiedano già una rappresentazione latente quasi perfetta del volto umano e delle sfumature paralinguistiche della voce. Tentare di modificare i pesi di questi giganti su un dataset piccolo come DOLOS rischierebbe solo di "dimenticare" questa conoscenza pregressa (*catastrophic forgetting*).

Il principio consiste quindi nel **congelare totalmente (100%) gli estrattori unimodali** e concentrare tutta la capacità di apprendimento sul **meccanismo di fusione multimodale**. La menzogna, infatti, spesso emerge dall'*incongruenza* tra i canali (es. una voce ferma accompagnata da una micro-espressione di tensione, o viceversa). Utilizzando meccanismi di *Cross-Attention*, il modello impara a interrogare il flusso visivo usando il flusso acustico come *Query* (e viceversa), allineando temporalmente i due segnali con un costo computazionale irrisorio.

*Spunti per varianti sul tema:*
* Sperimentare diverse strategie di fusione: *Co-Attention* simmetrica, *Bottleneck Attention* (usare pochi token latenti per far comunicare audio e video, stile architettura *Perceiver*), oppure *Gated Multimodal Units* (GMU).
* Estrarre le feature dai penultimi layer dei backbone invece che dall'ultimo, poiché i layer intermedi spesso conservano informazioni spaziali/acustiche più granulari e meno astratte.

#### Implementazione Pratica
* **Backbone Video (Congelato):** **DINOv2** o **SigLIP**. A differenza dei ViT classici addestrati per classificare oggetti generici, DINOv2 cattura dettagli geometrici e locali finissimi (ottimo per le micro-espressioni). Passate i 16 frame singolarmente nel modello con `torch.no_grad()` per ottenere una sequenza di 16 vettori di embedding.
  * *Download:* [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) oppure [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) (tramite libreria `transformers`).
* **Backbone Audio (Congelato):** **WavLM**. È l'evoluzione di Wav2Vec 2.0 sviluppata da Microsoft; durante il pre-training è stato addestrato anche su task di separazione del parlato e paralinguistica, rendendolo superiore nel catturare stress vocale, pitch ed esitazioni.
  * *Download:* [`microsoft/wavlm-base-plus`](https://huggingface.co/microsoft/wavlm-base-plus) (tramite `transformers`).
* **Modulo Addestrabile (PyTorch nativo):** 
  1. Due layer lineari (`nn.Linear`) per proiettare gli embedding di DINOv2 ($384$ dim) e WavLM ($768$ dim) nella stessa dimensione comune (es. $d_{\text{model}} = 256$).
  2. Un modulo `nn.MultiheadAttention` per calcolare la Cross-Attention tra la sequenza temporale video e quella audio, seguito da un `nn.TransformerEncoderLayer` e un classificatore MLP finale.
  3. **Trucco salva-tempo:** Poiché i backbone sono congelati, è possibile fare un passo di pre-estrazione salvando i tensori di output di DINOv2 e WavLM su disco (`.pt`). In questo modo l'addestramento della rete di fusione richiederà meno di 1 minuto per epoca.

---

### Paradigma 2: Spatio-Temporal Transformers con Fine-Tuning PEFT / LoRA --LORE

#### Spiegazione Teorica
A differenza del Paradigma 1 (che analizza i frame come immagini statiche separate e poi le unisce), questo approccio utilizza **Transformer Spazio-Temporali nativi**. Modelli come VideoMAE o ViViT non guardano il singolo frame, ma dividono l'intero clip video in "tubi" 3D (*3D tubelets*), applicando l'attenzione simultaneamente nello spazio e nel tempo fin dal primo layer.

Tuttavia, poiché questi modelli sono stati pre-addestrati su dataset di azioni macroscopiche (come *Kinetics-400*: persone che giocano a calcio, nuotano o ballano), le loro matrici di attenzione non sono immediatamente ottimizzate per notare un impercettibile battito di ciglia o una contrazione delle labbra. È quindi necessario adattare i pesi interni del backbone senza però causare overfitting o esaurire la memoria della GPU. 

La soluzione teorica è il **PEFT (Parameter-Efficient Fine-Tuning)**, nello specifico tramite **LoRA (Low-Rank Adaptation)**. LoRA ipotizza che il cambiamento dei pesi necessario per adattare un grande modello a un nuovo task abbia un "rango intrinseco basso". Invece di aggiornare una matrice di pesi $W \in \mathbb{R}^{d \times k}$, la si congela e si affiancano due matrici minuscole $A \in \mathbb{R}^{d \times r}$ e $B \in \mathbb{R}^{r \times k}$ (con rango $r \ll d, k$). In questo modo si aggiorna meno dell'1% dei parametri totali, modificando il comportamento interno del Transformer con un consumo minimo di VRAM.

*Spunti per varianti sul tema:*
* Confrontare diversi metodi PEFT oltre a LoRA, come **DoRA** (Weight-Decomposed Low-Rank Adaptation) o **AdaLoRA**, già integrati nella stessa libreria.
* Testare il posizionamento degli adapter: applicare LoRA solo ai layer di attenzione (`query`, `value`) oppure includere anche i layer *feed-forward* (MLP) degli ultimi blocchi del Transformer.
* Esplorare diverse tecniche di *Late Fusion* (media pesata dei logit, concatenazione degli stati nascosti prima del classificatore, o *Confidence-weighted fusion*).

#### Implementazione Pratica
* **Modello Video (con LoRA):** **VideoMAE** (Video Masked Autoencoder). È attualmente lo standard più solido su singola GPU per la comprensione video, molto più efficiente in memoria rispetto ai primi ViViT. Prende in input un tensore di forma `(Batch, 16, 3, 224, 224)`.
  * *Download:* [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) oppure la versione V2 [`OpenGVLab/VideoMAEv2-Base`](https://huggingface.co/OpenGVLab/VideoMAEv2-Base).
* **Modello Audio (con LoRA):** **HuBERT** oppure il solo encoder acustico di **Whisper**. HuBERT apprende unità acustiche discrete nascoste ed è un punto di riferimento nella letteratura sull'analisi delle emozioni e dell'inganno.
  * *Download:* [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960) oppure [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (usando `WhisperModel.encoder`).
* **Libreria per il Training:** **HuggingFace PEFT** (`pip install peft`).
  * Si avvolgono entrambi i modelli con `get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"]))`.
  * Per gestire la memoria su Colab/Laptop: usare un `batch_size = 2` o `4` in combinazione con `gradient_accumulation_steps = 8` e training in precisione mista (`fp16` o `bf16`).

---

### Paradigma 3: Architetture Convoluzionali 3D Moderne + Modellazione Sequenziale --FAB

#### Spiegazione Teorica
I Vision Transformer (Paradigmi 1 e 2) mancano di *inductive bias* (preconcetti strutturali): non sanno cosa sia il concetto di "vicinanza" tra pixel o tra istanti temporali contigui finché non lo imparano da enormi quantità di dati. Su dataset piccoli e rumorosi come DOLOS, le **Reti Convoluzionali (CNN)** offrono spesso una regolarizzazione naturale superiore grazie alla località dei filtri convoluzionali e all'invarianza alla traslazione.

Questo paradigma evolve la classica architettura *I3D + LSTM* utilizzando convoluzioni spazio-temporali moderne ed estremamente leggere per estrarre pattern di movimento locali su brevi finestre temporali, accoppiate a una rappresentazione tempo-frequenza (spettrogramma) per l'audio. L'audio non viene trattato come un'onda 1D, ma convertito in uno **Spettrogramma Mel 2D** (un'immagine in cui l'asse X è il tempo, l'asse Y è la frequenza percepita dall'orecchio umano e il colore è l'intensità). Questo permette di usare le più recenti architetture di visione artificiale anche per analizzare la voce (catturando visivamente le armoniche, i tremori nella frequenza fondamentale $F_0$ e i silenzi).

Infine, la dinamica a lungo raggio dell'intera clip viene delegata a un classificatore sequenziale dedicato (come una rete ricorrente bidirezionale o una rete convoluzionale temporale dilatata) che legge l'evoluzione delle feature dall'inizio alla fine della risposta del soggetto.

*Spunti per varianti sul tema:*
* Sostituire le classiche RNN (LSTM/GRU) nel classificatore finale con una **TCN (Temporal Convolutional Network)** o un blocco **Mamba (State Space Model)**, che gestiscono meglio sequenze lunghe senza soffrire di *vanishing gradient*.
* Dividere il video in brevi *chunk* sovrapposti (es. finestre di 8 frame con stride 4), estrarre un vettore per ogni finestra con la CNN 3D e passare la sequenza di finestre al modulo ricorrente.

#### Implementazione Pratica
* **Modello Video (CNN 3D / Ibrida):** **X3D** (Expand 3D) di Meta oppure **Swin3D**. X3D è una rete convoluzionale 3D progettata tramite espansione progressiva degli assi: offre performance superiori a I3D e SlowFast richiedendo circa 5 volte meno calcolo (FLOPs) e pochissima memoria.
  * *Download:* Disponibili nativamente in PyTorch senza dipendenze esterne complesse tramite [`torchvision.models.video`](https://pytorch.org/vision/stable/models.html#video-classification) (es. `swin3d_t(weights='DEFAULT')` o `mvit_v2_s`) oppure tramite **PyTorchVideo** (`torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)`).
* **Modello Audio (Spettrogramma + CNN 2D):** 
  1. Trasformare il file `.wav` in Mel-Spectrogram usando `torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)`.
  2. Passare lo spettrogramma (come immagine a 1 o 3 canali) dentro **ConvNeXt-Tiny**, una CNN pura modernizzata con i principi di design dei Transformer, leggera e potentissima.
  * *Download:* [`timm.create_model('convnext_tiny.fb_in22k', pretrained=True)`](https://huggingface.co/timm/convnext_tiny.fb_in22k) tramite la libreria `timm` (`pip install timm`).
* **Modulo Sequenziale Addestrabile:**
  * Estrarre le sequenze di feature temporali da X3D e ConvNeXt, concatenarle lungo l'asse delle feature e darle in pasto a una **Bi-GRU** (`nn.GRU(input_size, hidden_size=128, bidirectional=True, batch_first=True)`) con un elevato *Dropout* ($p=0.4$ o $0.5$) per contrastare l'overfitting, seguita da un layer lineare per la predizione binaria (Verità / Bugia).

---

## Parte 3: Riepilogo Comparativo per il Paper

Questa struttura vi permetterà di presentare nella relazione finale uno studio comparativo completo che esplora le tre grandi famiglie del Deep Learning multimodale:

| Paradigma | Filosofia di Base | Backbone Video | Backbone Audio | Modulo Addestrato |
| :--- | :--- | :--- | :--- | :--- |
| **1. PECL-Style** | Estrazione statica congelata + Fusione profonda | `dinov2-small` / `siglip` | `wavlm-base-plus` | Cross-Modal Attention Transformer |
| **2. End-to-End PEFT** | Adattamento interno a basso rango (LoRA) | `videomae-base` | `hubert-base` / `whisper` | Matrici LoRA ($r=8$) + Late Fusion |
| **3. 3D-CNN + Spec** | Inductive bias convoluzionale + Ricorrenza | `x3d_m` / `swin3d_t` | Mel-Spec + `convnext_tiny` | Bi-GRU / TCN (Temporal ConvNet) |