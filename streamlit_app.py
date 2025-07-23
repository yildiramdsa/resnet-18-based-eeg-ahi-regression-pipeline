import streamlit as st

PAGE_TITLE = "ResNet-18-Based EEG AHI Regression Pipeline"

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide"
)

def show_abstract():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Abstract")
        st.markdown("""
        Sleep apnea is a prevalent sleep disorder affecting millions worldwide, with significant cardiovascular and cognitive consequences. We present a <span class="inline-badge">deep learning pipeline for automated Apnea-Hypopnea Index (AHI) estimation from single-channel EEG recordings using spectrogram-based analysis</span>.
        
        Our **approach utilizes the Sleep Heart Health Study (SHHS-1) dataset, processing C3/A2 EEG signals into 30-second spectrogram windows with comprehensive quality control**. We implement a **ResNet-18 architecture with transfer learning from ImageNet pretraining, optimized for continuous AHI regression** rather than categorical classification. The model incorporates **subject-stratified cross-validation, learning rate warmup, early stopping, and data augmentation** to ensure robust generalization.
        
        Results demonstrate moderate performance with a **Root Mean Square Error (RMSE) of 6.8 events/hour** and a **Pearson correlation of 0.76** on held-out validation data. While systematic biases exist (**overprediction at low AHI, underprediction at high AHI**), the model provides a **clinically relevant severity ranking suitable for screening applications**. Our work establishes the feasibility of automated sleep apnea assessment from standard EEG recordings.

        **Keywords**: Sleep apnea, EEG spectrograms, deep learning, ResNet-18, AHI regression
        """, unsafe_allow_html=True)

def show_introduction():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Introduction")
        st.markdown("""
        **Sleep apnea** is a common **sleep disorder in which the upper airway repeatedly narrows or collapses during sleep, causing pauses in breathing (apneas) or shallow breathing episodes (hypopneas)**. These interruptions fragment sleep and reduce blood oxygen levels, leading to **daytime fatigue**, **morning headaches**, **impaired concentration**, and an **increased risk of cardiovascular and metabolic disorders**. The **severity of sleep apnea** is quantified by the <span class="inline-badge">Apnea-Hypopnea Index (AHI)</span>, which measures the <span class="inline-badge">number of breathing events per hour of sleep</span>. Accurate AHI assessment is crucial for clinical decision-making.

        The high prevalence of sleep apnea underscores the need for more accessible diagnostic tools. In Canada, an estimated 6.4% of adults received a professional diagnosis of sleep apnea in 2016-2017, while population-based assessments suggest that **nearly 28.1% of Canadians aged 45 to 85 years have moderate to severe obstructive sleep apnea (AHI ≥15 events/hour)** based on STOP-BANG screening (Statistics Canada, 2018; Canadian Longitudinal Study on Aging Team, 2024).

        Electroencephalography (EEG) offers a promising solution for the automated assessment of sleep apnea. As a non-invasive neuroimaging technique that records electrical activity in the brain through electrodes placed on the scalp, EEG captures the synchronized firing of neurons, producing characteristic waveforms that vary with different states of consciousness and neurological conditions. During sleep, **EEG recordings reveal distinct patterns associated with different sleep stages and can detect disruptions caused by sleep disorders such as apnea**. The temporal resolution of EEG (milliseconds) makes it ideal for capturing rapid changes in brain activity that occur during sleep-wake transitions and respiratory events, providing a rich source of information for automated analysis.

        However, **raw EEG signals present significant challenges for automated analysis**. They are **highly sensitive to artifacts (such as muscle activity, eye movements, and electrode noise), require extensive preprocessing, and contain complex temporal dependencies that are challenging for deep learning models to learn directly**. <span class="inline-badge">Spectrograms</span> **address these limitations by** <span class="inline-badge">transforming the time-domain signal into a time-frequency representation</span> **that preserves both temporal and spectral information.** This transformation makes sleep apnea-related patterns more visually apparent and computationally tractable, effectively highlighting frequency bands associated with different sleep stages (delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-13 Hz, beta: 13-30 Hz) and revealing disruptions in these patterns caused by apnea events. Additionally, spectrograms are more robust to noise and artifacts, as the frequency domain representation naturally filters out many types of interference.

        We built our pipeline on <span class="inline-badge">ResNet-18</span> to treat **spectrograms as 2D images**—with **residual connections that preserve gradient flow, an 18-layer depth that balances capacity and regularization, and ImageNet pretrained weights that jump-start AHI regression learning**. Its lightweight design also ensures the speed needed for real-time clinical use.

        Our approach performs <span class="inline-badge">continuous AHI regression directly from windowed EEG spectrogram images</span>, rather than using categorical classification, which provides more precise severity estimates that are clinically valuable. The pipeline incorporates **subject-stratified training and validation splits** to prevent data leakage, **comprehensive quality control** to ensure reliable spectrogram generation, **learning-rate warmup and early stopping** to optimize training, and **data augmentation** to improve generalization. **Window-level predictions are aggregated at the subject level** to yield per-individual AHI estimates suitable for clinical screening and research applications.
        """, unsafe_allow_html=True)

def show_literature_review():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Literature Review")
        st.markdown("""
        Building on the need for scalable, automated estimation of sleep apnea severity from EEG spectrograms, recent work in automated sleep analysis can be categorized into three complementary areas: **sleep stage scoring**, **disorder classification**, and **integrated multi-task frameworks**.
    
        <span class="inline-badge">Sleep Stage Scoring with Spectrogram-Based Deep Learning</span> Spectrograms of EEG preserve both time and frequency information, letting convolutional networks learn patterns that distinguish sleep stages. Li et al. (2022) introduced EEGSNet, a hybrid CNN-BiLSTM model that achieved over 94% accuracy on the Sleep-EDF-8 dataset and strong agreement (κ > 0.77) on several public cohorts by combining learned spectral features with temporal context. Likewise, Tsinalis et al. (2016) demonstrated that an end-to-end CNN trained on single-channel EEG spectrograms can achieve balanced F1-scores of approximately 0.81 without manual feature design, indicating that deep filters naturally capture stage-specific rhythms.
    
        <span class="inline-badge">Disorder Detection from Physiological Signals</span> Beyond staging, identifying specific sleep pathologies enables the delivery of targeted care. Zhuang and Ibrahim (2022) developed DL-R, a multi-channel CNN that utilizes raw EEG, EMG, ECG, and EOG signals to classify eight sleep disorders with a sensitivity and specificity of over 95%. Gawhale et al. (2023) extracted deep features from EEG spectrograms and fed them to an ensemble classifier, achieving an overall accuracy of 96.8%, which highlights the feasibility of real-time disorder screening on lightweight devices.

        <span class="inline-badge">Ensemble, Multimodal, and Multi-Task Strategies</span> To combat data imbalance and leverage diverse signals, ensemble and multi-task methods have emerged as effective solutions. Monowar et al. (2025) combined multiple learners with SMOTE-based augmentation to achieve 99.5% cross-validated accuracy for disorder detection, outperforming standalone models. Khanmohmmadi et al. (2025) proposed a multi-task CNN optimized via genetic and Q-learning that jointly predicts sleep deprivation and disorder labels, achieving 98% accuracy by sharing intermediate representations. Cheng et al. (2023) fused parallel CNNs on EEG, ECG, and EMG to perform simultaneous sleep stage and disorder classification, reporting 94.3% staging accuracy and 99.1% disorder accuracy—underscoring the benefit of integrated, multimodal modelling.

        While these classification advances are impressive, most **focus on categorical labels rather than continuous severity measures**.
        
        <span class="inline-badge">Reproduction and Evaluation of Tanci & Hekim (2025) Findings</span> Tanci and Hekim (2025) proposed a **streamlined, single-channel pipeline for four-class sleep apnea staging using 30-second C3-A2 EEG windows**. Each window was converted into a **Hann-windowed STFT spectrogram** and fed into three deep models—**ResNet-64, YOLOv5, and YOLOv8—for classification as healthy (AHI < 5), mild (5-14.9 events/hr), moderate (15-29.9 events/hr), or severe (≥ 30 events/hr)**. Evaluated on **25 PhysioBank ATM subjects**, **YOLOv8 led with 93.7% correct classification, slightly outperforming ResNet-64 (93.0%) and significantly surpassing YOLOv5 (88.2%)**. By focusing solely on the C3-A2 channel, the authors minimized preprocessing and model complexity while still capturing critical neural features, demonstrating a balance of accuracy, speed, and practicality for potential real-time clinical use.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.image("assets/tanci_hekim_pipeline.png", caption='Sleep-apnea staging pipeline. Raw C3-A2 EEG signals were segmented into 30-second windows, converted to STFT spectrograms, processed by deep learning models (ResNet-64, YOLOv5, YOLOv8), and classified into four AHI-based severity classes. *From "Classification of sleep apnea syndrome using the spectrograms of EEG signals and YOLOv8 deep learning model," by K. Tanci & M. Hekim, 2025, PeerJ Computer Science, 11, e2718. https://doi.org/10.7717/peerj-cs.2718. Copyright 2025 by Tanci and Hekim.*', use_container_width=True)
            st.image("assets/tanci_hekim_eeg_spectrograms.png", caption='EEG waveforms and corresponding spectrograms for each sleep-apnea severity level. (A) Mild, (B) Moderate, (C) Severe, and (D) Healthy; left panels show 30-second C3-A2 EEG segments, right panels display their STFT spectrograms. *From "Classification of sleep apnea syndrome using the spectrograms of EEG signals and YOLOv8 deep learning model," by K. Tanci & M. Hekim, 2025, PeerJ Computer Science, 11, e2718. https://doi.org/10.7717/peerj-cs.2718. Copyright 2025 by Tanci and Hekim.*', use_container_width=True)
        st.markdown("""
        We replicated Tanci & Hekim's pipeline in four steps: **segment C3-A2 EEG into 30-s windows**, **apply a Hann-window STFT (256-point FFT, 50% overlap) and convert to dB**, **discard any flat epochs**, and **export 150-dpi Viridis PNGs**. We then ran these images through **ResNet-64**, **YOLOv5**, and **YOLOv8** and **compared their accuracy on a balanced hold-out test set of 400 samples (100 per class)**.
        
        <span class="inline-badge">ResNet64 Performance Analysis</span> **ResNet64** achieved a **balanced accuracy of 85.50%**, with **macro-averaged precision of 0.8703**, **recall of 0.8550**, and **F1-score of 0.8555**. The model demonstrated strong discriminative capabilities across all classes, as evidenced by high **AUC values ranging from 0.95 to 0.99** and **Average Precision values from 0.90 to 0.97**.

        <span class="inline-badge">Severe Apnea</span> The model achieved exceptional performance for severe cases with **precision = 0.91**, **recall = 0.92**, and **F1-score = 0.92**. The confusion matrix reveals that **92 out of 100 true severe cases were correctly classified (92.0% accuracy)**, with only 7% misclassified as mild and 1% as moderate. The ROC curve shows an **AUC of 0.99**, and the Precision-Recall curve demonstrates an **Average Precision of 0.97**, indicating near-perfect discriminative power for this critical severity level.

        <span class="inline-badge">Normal Cases</span> The normal class showed excellent performance with perfect **precision (1.00)**, meaning no other classes were misclassified as normal. However, the **recall was 75% (75 out of 100 true normal cases correctly identified)**, with 9% misclassified as mild, 14% as moderate, and 2% as severe. The ROC curve achieved an **AUC of 0.99**, and the Precision-Recall curve shows an **Average Precision of 0.97**, highlighting the model's strong ability to avoid false positives for normal cases.

        <span class="inline-badge">Moderate Apnea</span> Moderate cases showed balanced performance with **precision = 0.82**, **recall = 0.81**, and **F1-score = 0.81**. The confusion matrix indicates **81 out of 100 true moderate cases were correctly classified**, with 15% misclassified as mild and 4% as severe. The ROC curve shows an **AUC of 0.95**, and the Precision-Recall curve demonstrates an **Average Precision of 0.90**, reflecting the model's solid but slightly lower discriminative power for this intermediate severity level.

        <span class="inline-badge">Mild Apnea</span> Mild cases showed the most challenging performance with **precision = 0.75**, **recall = 0.94**, and **F1-score = 0.84**. While the model correctly identified **94 out of 100 true mild cases (high recall)**, it suffered from false positives, where 15 moderate and 9 normal cases were incorrectly predicted as mild. The ROC curve shows an **AUC of 0.97**, and the Precision-Recall curve demonstrates an **Average Precision of 0.92**, indicating good discriminative power but challenges in maintaining high precision.

        **ResNet64 demonstrates robust performance across all sleep apnea severity classes, with particularly strong performance for severe and normal cases.** The model's ability to achieve high AUC values (0.95-0.99) across all classes indicates excellent discriminative capabilities, while the varying precision-recall trade-offs reflect the inherent challenges in distinguishing between adjacent severity categories, particularly between mild and moderate cases.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/ResNet64_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/ResNet64_roc_curves_balanced.png")
        with col3:
            st.image("assets/ResNet64_pr_curves_balanced.png")
        st.markdown("""
        <span class="inline-badge">YOLOv5 Performance Analysis</span> **YOLOv5** outperformed ResNet64 with **balanced accuracy of 88.75%**, **macro precision of 0.8972**, **recall of 0.8875**, and **F1 of 0.8887**. The model demonstrated exceptional discriminative capabilities across all classes, as evidenced by high **AUC values ranging from 0.98 to a perfect 1.00** and robust **Average Precision values from 0.95 to 0.99**.

        <span class="inline-badge">Normal Cases</span> YOLOv5 achieved near-perfect performance for normal cases with **precision = 0.99**, **recall = 0.85**, and **F1-score = 0.91**. The confusion matrix reveals that **85 out of 100 true normal cases were correctly classified**, with only 9% misclassified as mild, 4% as moderate, and 2% as severe. The ROC curve shows a perfect **AUC of 1.00**, and the Precision-Recall curve demonstrates an **Average Precision of 0.99**, indicating exceptional discriminative power and minimal false positives for normal cases.

        <span class="inline-badge">Severe Apnea</span> The model showed robust performance for severe cases with **precision = 0.94**, **recall = 0.88**, and **F1-score = 0.91**. The confusion matrix indicates **88 out of 100 true severe cases were correctly classified**, with only 9% misclassified as mild and 3% as moderate. The ROC curve shows an **AUC of 0.99**, and the Precision-Recall curve demonstrates an **Average Precision of 0.98**, highlighting the model's strong ability to identify severe cases with high confidence.

        <span class="inline-badge">Mild Apnea</span> Mild cases showed the highest **recall (0.96)** among all classes, with **precision = 0.79** and **F1-score = 0.86**. The confusion matrix reveals that **96 out of 100 true mild cases were correctly identified**, demonstrating excellent sensitivity. However, the model suffered from false positives, where 26 cases (10 moderate, 9 normal, 7 severe) were incorrectly predicted as mild. The ROC curve shows an **AUC of 0.98**, and the Precision-Recall curve demonstrates an **Average Precision of 0.95**, indicating good discriminative power but challenges in maintaining high precision.

        <span class="inline-badge">Moderate Apnea</span> Moderate cases showed balanced performance with **precision = 0.88**, **recall = 0.86**, and **F1-score = 0.87**. The confusion matrix indicates **86 out of 100 true moderate cases were correctly classified**, with 10% misclassified as mild and 4% as severe. Additionally, 12 cases were falsely predicted as moderate when they belonged to other classes. The ROC curve shows an **AUC of 0.98**, and the Precision-Recall curve demonstrates an **Average Precision of 0.96**, reflecting solid discriminative power for this intermediate severity level.

        **YOLOv5 demonstrates superior performance compared to ResNet64, achieving higher balanced accuracy (88.75% vs 85.50%) and more consistent performance across all classes.** The model's ability to achieve a perfect AUC (1.00) for normal cases and high AUC values (0.98-0.99) across all classes indicates excellent discriminative capabilities. While mild cases show the highest recall, the precision-recall trade-offs reflect the inherent challenges in distinguishing between adjacent severity categories, particularly the tendency to over-classify mild cases.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/YOLOv5_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/YOLOv5_roc_curves_balanced.png")
        with col3:
            st.image("assets/YOLOv5_pr_curves_balanced.png")
        st.markdown("""
        <span class="inline-badge">YOLOv8 Performance Analysis</span> **YOLOv8**, contrary to the original report, yielded a lower **balanced accuracy of 72.25%**, with **macro precision of 0.7409**, **recall of 0.7225**, and **F1 score of 0.7247**. The model demonstrated moderate discriminative capabilities across all classes, as evidenced by **AUC values ranging from 0.89 to 0.95** and **Average Precision values from 0.78 to 0.91**, significantly underperforming compared to both ResNet64 and YOLOv5.

        <span class="inline-badge">Normal Cases</span> YOLOv8 achieved the highest precision for normal cases with **precision = 0.93**, **recall = 0.66**, and **F1-score = 0.77**. The confusion matrix reveals that **66 out of 100 true normal cases were correctly classified**, with 16% misclassified as mild and 12% as moderate. The ROC curve shows an **AUC of 0.95**, and the Precision-Recall curve demonstrates an **Average Precision of 0.91**, indicating good discriminative power for normal cases despite the lower recall.

        <span class="inline-badge">Severe Apnea</span> The model showed the highest recall for severe cases with **precision = 0.72**, **recall = 0.80**, and **F1-score = 0.76**. The confusion matrix indicates **80 out of 100 true severe cases were correctly classified**, with 12% misclassified as mild and 8% as moderate. Notably, no severe cases were misclassified as normal. The ROC curve shows an **AUC of 0.91**, and the Precision-Recall curve demonstrates an **Average Precision of 0.83**, highlighting the model's ability to identify most of the severe cases.

        <span class="inline-badge">Moderate Apnea</span> Moderate cases showed poor performance with **precision = 0.68**, **recall = 0.71**, and **F1-score = 0.69**. The confusion matrix reveals that **71 out of 100 true moderate cases were correctly classified**, with 13% misclassified as mild and 13% as severe. Additionally, 13 cases were falsely predicted as moderate when they belonged to other classes. The ROC curve shows an **AUC of 0.90**, and the Precision-Recall curve demonstrates an **Average Precision of 0.78**, reflecting the model's challenges in distinguishing moderate cases from adjacent categories.

        <span class="inline-badge">Mild Apnea</span> Mild cases showed the poorest performance with **precision = 0.64**, **recall = 0.72**, and **F1-score = 0.68**. The confusion matrix indicates **72 out of 100 true mild cases were correctly classified**, with 14% misclassified as moderate and 12% as severe. The model also suffered from false positives, where 28 cases were incorrectly predicted as mild. The ROC curve shows the lowest **AUC of 0.89**, and the Precision-Recall curve demonstrates an **Average Precision of 0.81**, indicating the weakest discriminative power among all classes.

        **YOLOv8 demonstrates significantly inferior performance compared to both ResNet64 and YOLOv5, achieving the lowest balanced accuracy (72.25% vs 85.50% and 88.75%) and showing substantial challenges in distinguishing between adjacent severity categories.** The model's AUC values (0.89-0.95) and Average Precision values (0.78-0.91) are notably lower than the other models, particularly for mild and moderate cases.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/YOLOv8_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/YOLOv8_roc_curves_balanced.png")
        with col3:
            st.image("assets/YOLOv8_pr_curves_balanced.png")
        st.markdown("""
        Tanci and Hekim's (2025) approach has several **design flaws**. First, **labelling every 30-s window by the subject's overall AHI—even when no apnea occurs**—injects noisy, misleading targets. **Randomly splitting windows into train/test sets also allows the same subject to appear in both sets**, inflating performance by memorizing person-specific patterns. **Fixed-epoch training without learning-rate scheduling or early stopping** risks under- or overfitting, and the **lack of quality-control filtering** leaves many meaningless spectrograms in the data. Finally, **reducing AHI to four categories** overlooks its continuous nature, potentially masking significant differences in severity. These choices likely lead to an **overly optimistic 93.7% accuracy**, which will not generalize well to new patients.
        
        Instead of classifying windows, we **regress continuous AHI from the C3/A2 EEG of 100 SHHS-1 subjects**. We **split recordings into 30-s windows (50% overlap)**, **filtered out low- and high-variance epochs (5-200 µV)**, and **converted the rest into 224×224 Viridis spectrograms via a 256-point Hann STFT with cohort-wide normalization**. **An ImageNet-pretrained ResNet-18 with a custom regression head was fine-tuned under subject-stratified 5-fold cross-validation, using learning-rate warmup, early stopping, and data augmentation.** We **averaged window predictions per subject to get final AHI scores** and assessed performance with RMSE, MAE, and Pearson correlation, yielding precise, clinically relevant estimates that generalize to new patients.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.image("assets/ahi_regression_pipeline.png", caption='Overview of the ResNet-18-based EEG AHI regression pipeline. Raw C3-A2 EEG signals were segmented into 30-second windows, transformed into STFT spectrograms, processed by a ResNet-18 regression model, and yielded continuous AHI predictions.', use_container_width=True)

def show_methods():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Materials & Methods")
        st.markdown("""
        <span class="inline-badge">Dataset</span> We utilized the **Sleep Heart Health Study 1 (SHHS-1) dataset**, which collected overnight sleep recordings from **6,441 adults (40 years or older)** between 1995 and 1998. Participants came from nine extensive health studies (e.g., Framingham Offspring, ARIC, CHS, Strong Heart), and some sites intentionally included more snorers to capture a range of sleep-breathing issues. All recordings were done at home by trained technicians using a complete sleep montage: **two EEG channels (C3/A2, C4/A1 at 125 Hz)**, eye movements (EOG at 50 Hz), muscle tone (EMG at 125 Hz), breathing effort (plethysmography at 10 Hz), airflow (thermocouple at 10 Hz), blood oxygen (pulse oximetry at 1 Hz), heart rhythm (ECG at 125 Hz), plus body-position and light sensors. For our model, we automatically select the **C3/A2 EEG channel to create spectrograms at 125 Hz, preserving both time and frequency details intact**.

        From the complete set of 6,441 recordings, we selected **100 subjects evenly distributed across severity levels—49 healthy (AHI < 5), 30 mild (AHI 5-14.9), 14 moderate (AHI 15-29.9), and 7 severe (AHI ≥ 30)**. This sample reflects the **typical right-skewed distribution of AHI in the general population**, with most people having low scores and fewer at the severe end.
        """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("assets/ahi_distribution_histogram.png", use_container_width=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("""
        <span class="inline-badge">Spectrogram Generation & Quality Control</span> We **converted raw EEG recordings into standardized 224×224-pixel spectrogram images** using a multi-step pipeline tailored for clinical applications. First, we **segmented the C3/A2 EEG signal into 30-second windows with 50% overlap** (per AASM guidelines) to capture transitions in breathing events. **Each window underwent a short-time Fourier transform (STFT) with a 256-point FFT and a 128-point hop length**, producing smooth time-frequency representations. We then **applied global contrast normalization—using fixed decibel limits (vmin = -53.68 dB, vmax = 32.78 dB) — computed across the entire cohort**—to ensure consistent colour scaling.
        
        For quality control, we **discarded any epoch with abnormally low variance (standard deviation ≤ 5.0)**, indicating a flat or disconnected channel, **as well as epochs with extremely high variance (standard deviation ≥ 200.0)**, which reflects movement or sensor artifacts. **Surviving windows were rendered as PNG images using the Viridis colormap, saved without padding or borders** (see example spectrograms below), and named `{subject_id}_{window_index:04d}.png` within per-subject directories (`resnet224_color/{subject_id}/`). We also **recorded metadata, including file path and true AHI**, in a master `window_metadata.csv`.
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image("assets/example_spectrogram_normal.png", use_container_width=True)
        with col2:
            st.image("assets/example_spectrogram_mild.png", use_container_width=True)
        with col3:
            st.image("assets/example_spectrogram_moderate.png", use_container_width=True)
        with col4:
            st.image("assets/example_spectrogram_severe.png", use_container_width=True)
        st.markdown("""
        To prevent data leakage, we **adopted a five-fold stratified group cross-validation scheme**: subjects, rather than individual windows, were assigned to folds so that each fold contained approximately 31,000-36,000 high-quality spectrograms and maintained a balanced distribution of apnea severities. Overall, this process yielded **over 166,000 spectrogram windows across 100 subjects**.

        <span class="inline-badge">ResNet-18 Model Architecture & Training</span> We **fine-tuned a ResNet-18 backbone, initialized with ImageNet weights, for continuous AHI regression on 224×224 RGB EEG spectrograms by replacing its final classifier with a single-node linear output**. The **input spectrograms pass through 18 residual layers organized into four blocks**, and a **global average pooling layer reduces spatial dimensions to a 512-dimensional feature vector, which our regression head then maps to an AHI score**. To balance feature reuse and task adaptation, we **employed differential learning rates: a low rate (1 × 10⁻⁵) for the pre-trained backbone and a higher rate (1 × 10⁻⁴) for the new regression head**. Training was driven by the Smooth L1 (Huber) loss, optimized with the Adam algorithm and weight decay, while gradient clipping (norm ≤ 1.0) ensured stable updates.

        Our training schedule began with a **five-epoch warmup that linearly increased the learning rate from zero to its target**, followed by a **ReduceLROnPlateau strategy that halved the rate after five epochs of stagnant validation loss**. We also used **early stopping—halting training after ten epochs without improvement**—to avoid overfitting. To improve generalization, we applied **random horizontal flips and colour jitter (±20% brightness and contrast)** to the spectrograms. During inference, **window-level predictions were averaged per subject to yield a single AHI estimate**, reflecting the clinical need for a single severity score.

        All **spectrograms were normalized using ImageNet's mean and standard-deviation statistics and labelled with each subject's overnight AHI**. We **trained for up to 30 epochs with a batch size of 16 under a five-fold, subject-stratified cross-validation scheme—80 subjects for training and 20 for validation in each fold**—to prevent data leakage and ensure realistic performance on unseen individuals. We evaluated our model by **computing the subject-level mean squared error (MSE) and mean absolute error (MAE) between the averaged predictions and the true AHI values**.
        """, unsafe_allow_html=True)

def show_results():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Results & Discussion")
        st.markdown("""
        Our ResNet-18-based AHI regression model achieved moderate performance on the held-out validation set, demonstrating the feasibility of estimating sleep apnea severity from EEG spectrograms using automated methods. The model was evaluated using **subject-level aggregation, where window-level predictions were averaged per individual** to yield clinically relevant **per-subject AHI estimates**.
    
        <span class="inline-badge">Model Performance Metrics</span> The model achieved a **Root Mean Square Error (RMSE) of 6.8 events/hour** and a **Mean Absolute Error (MAE) of 5.2 events/hour** on the validation set. These metrics indicate that, on average, the model's predictions deviate from true AHI values by approximately 6.8 events/hour in terms of squared error and 5.2 events/hour in absolute terms. The **Pearson correlation coefficient of 0.76** demonstrates moderate linear agreement between predicted and true AHI values, indicating room for improvement in capturing the underlying severity relationships.
    
        The scatter plot below reveals several important characteristics of the model's performance across the AHI severity spectrum. The model demonstrates moderate correlation with true AHI values, with predictions showing **considerable spread around the perfect prediction line**.
        """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("assets/ahi_regression_scatter_plot.png", use_container_width=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("""
        The model shows **distinct performance patterns across different AHI ranges**:

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Low AHI (0-5)**: Most points lie well above the 45° line, with predictions in the 5-12 events/hr range for true AHI values of 0-5. The model systematically overestimates very low-severity cases by approximately 4-8 events/hr, indicating difficulty in distinguishing healthy from very mild apnea.

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Mild AHI (5-15)**: Predictions cluster around the mid-range (8-18 events/hr), even when the true AHI spans 5-15. Agreement is better than at the extremes, but there's still high variability—some true values near 10-15 events/hr are underpredicted by ~2-4 events/hr, while others are slightly overpredicted.

        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Moderate to Severe AHI (15-35)**: All points for true AHI in this band fall below the identity line, with predicted values 5-8 events/hr lower than the true score. This consistent underprediction suggests that the model is conservative for higher-severity cases, likely a consequence of the sparse training examples above AHI 15.

        <span class="inline-badge">Model Robustness</span> By using **subject-stratified cross-validation**—where all data from a given individual lives entirely in either the training or validation fold—we ensure that our **performance metrics truly reflect the model's ability to generalize to new subjects** rather than simply memorizing person-specific patterns. The fact that **each fold yields similar accuracy and error statistics** highlights the **model's capacity to extract EEG spectrogram features that reliably track apnea severity across diverse individuals**. This robustness suggests that the learned representations capture physiologically meaningful signal patterns rather than idiosyncratic noise, a critical property for any clinical deployment aiming to assess patients unseen during training.

        <span class="inline-badge">Limitations and Future Work</span> Several limitations should be considered when interpreting these results. First, our model was trained on **only 100 subjects**, which limits its ability to capture the full diversity of sleep-apnea presentations and EEG signatures. Second, we observed a systematic bias: the model overestimates very low AHI values (0-5) by approximately 4-8 events per hour and underestimates moderate to severe cases (15-35) by a similar margin, suggesting that **extremes are underrepresented in the training data**. To address these issues, future work should **expand and diversify the dataset** by recruiting subjects across a broader range of ages, comorbidities, and sleep-disorder severities, and **integrate multimodal physiological signals**, such as ECG, respiratory effort, and oximetry, alongside EEG spectrograms to improve sensitivity at both ends of the severity spectrum. Additionally, exploring **ensemble architectures** that combine complementary modelling approaches (for example, convolutional, transformer, and recurrent networks) may help reduce systematic biases and enhance overall accuracy. Despite these limitations, our results highlight the **promise of an automated, scalable approach to assessing sleep-apnea severity using standard EEG recordings**, offering a valuable foundation for both clinical screening and large-scale research applications.
        """, unsafe_allow_html=True)

def show_conclusion():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Conclusion")
        st.markdown("""
        This study demonstrates the feasibility of <span class="inline-badge">automated sleep apnea severity assessment using EEG spectrograms and deep learning</span>, achieving **moderate performance (RMSE: 6.8 events/hour, Pearson correlation: 0.76)** for scalable screening from standard EEG recordings. Although systematic biases exist across the AHI spectrum, the modular and scalable design of our pipeline enables easy adaptation to diverse datasets and clinical settings, supporting future integration of multi-modal physiological signals and real-time clinical applications.

        Key contributions of this work include the development of a <span class="inline-badge">robust preprocessing pipeline with stringent quality control for EEG spectrogram generation</span>, the application of <span class="inline-badge">transfer learning using a ResNet-18 architecture optimized for medical imaging</span>, and the implementation of <span class="inline-badge">continuous AHI regression for precise, subject-level estimation of sleep apnea severity</span>. The modular and scalable design of the pipeline ensures <span class="inline-badge">adaptability to diverse datasets</span> and facilitates <span class="inline-badge">integration into various clinical and research environments</span>.

        Future work should focus on **expanding training datasets**, **incorporating multi-modal physiological signals**, and **optimizing for real-time clinical applications**.

        **Code Availability:** The complete implementation, including data preprocessing, model training, and evaluation pipelines, is available at: [https://github.com/yildiramdsa/resnet-18-based-eeg-ahi-regression-pipeline](https://github.com/yildiramdsa/resnet-18-based-eeg-ahi-regression-pipeline)
        """, unsafe_allow_html=True)

def show_references():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("References")
        st.markdown("""
        Canadian Longitudinal Study on Aging Team. (2024). Prevalence and regional distribution of obstructive sleep apnea in Canada: Analysis from the Canadian Longitudinal Study on Aging. *Canadian Journal of Public Health, 115*(6), 970-979. https://doi.org/10.17269/s41997-024-00911-8

        Gawhale, S., Upasani, D. E., Chaudhari, L., Khankal, D. V., Kumar, J. R., & Upadhye, V. A. (2023). EEG signal processing for the identification of sleeping disorder using hybrid deep learning with ensemble machine learning classifier. *International Journal of Intelligent Systems and Applications in Engineering, 11*(10 Suppl.), 113-129. https://ijisae.org/index.php/IJISAE/article/view/3239

        Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., … Stanley, H. E. (n.d.). *University College Dublin Sleep Apnea Database (UCDDB)* [Data set]. PhysioNet. https://archive.physionet.org/physiobank/database/ucddb/

        Khanmohmmadi, S., Khatibi, T., Tajeddin, G., Akhondzadeh, E., & Shojaee, A. (2025). Revolutionizing sleep disorder diagnosis: A multi-task learning approach optimized with genetic and Q-learning techniques. *Scientific Reports, 15*, Article 16603. https://doi.org/10.1038/s41598-025-01893-4

        Li, C., Qi, Y., Ding, X., Zhao, J., Sang, T., & Lee, M. (2022). A deep learning method approach for sleep stage classification with EEG spectrogram. *International Journal of Environmental Research and Public Health, 19*(10), 6322. https://doi.org/10.3390/ijerph19106322

        Monowar, M. M., Nobel, S. M. N., Afroj, M., Hamid, M. A., Uddin, M. Z., Kabir, M. M., & Mridha, M. F. (2025). Advanced sleep disorder detection using multi-layered ensemble learning and advanced data balancing techniques. *Frontiers in Artificial Intelligence, 7*, 1506770. https://doi.org/10.3389/frai.2024.1506770

        National Sleep Research Resource. (2025). *Sleep Heart Health Study polysomnography EDF files* [Data set]. https://sleepdata.org/datasets/shhs/files/polysomnography/edfs

        Statistics Canada. (2018). *Sleep apnea in Canada, 2016 and 2017* [Health fact sheet]. Government of Canada. https://www150.statcan.gc.ca/n1/pub/82-625-x/2018001/article/54979-eng.htm

        Tanci, M., & Hekim, M. (2025). Classification of sleep apnea syndrome using the spectrograms of EEG signals and YOLOv8 deep learning model. *PeerJ Computer Science, 11*, e2718. https://doi.org/10.7717/peerj-cs.2718

        Tsinalis, O., Matthews, P. M., Guo, Y., & Zafeiriou, S. (2016). Automatic sleep stage scoring with single-channel EEG using convolutional neural networks. *arXiv*. https://doi.org/10.48550/arXiv.1610.01683

        Wara, T. U., Fahad, A. H., Das, A. S., & Shawon, M. M. H. (2025). A systematic review on sleep stage classification and sleep disorder detection using artificial intelligence. *arXiv*. https://doi.org/10.48550/arXiv.2405.11008

        Zhuang, D., & Ibrahim, A. K. (2022). A machine learning approach to automatic classification of eight sleep disorders. *arXiv*. https://doi.org/10.48550/arXiv.2204.06997
        """)

def show_terminology():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Key Terms")
        st.markdown("""
        <span class="inline-badge">AHI (Apnea-Hypopnea Index)</span> A measure of sleep apnea severity that counts the number of breathing pauses or shallow breathing episodes per hour of sleep. Higher numbers mean more severe sleep apnea.
    
        <span class="inline-badge">Alpha Waves</span> Brain waves that occur when you're relaxed but awake, typically between 8-13 Hz. They're important for identifying sleep stages.
    
        <span class="inline-badge">Apnea</span> A complete pause in breathing during sleep that lasts at least 10 seconds.
    
        <span class="inline-badge">Artifact</span> Unwanted signals in EEG recordings caused by muscle movement, eye blinking, or equipment problems that can interfere with analysis.
    
        <span class="inline-badge">Beta Waves</span> Fast brain waves (13-30 Hz) that occur during active thinking and alertness.
    
        <span class="inline-badge">C3/A2</span> A specific EEG electrode placement that records brain activity from the central part of the scalp. This is the primary channel used in this project.
    
        <span class="inline-badge">Classification</span> A type of prediction that puts data into categories (like normal/mild/moderate/severe) rather than estimating specific numbers.
    
        <span class="inline-badge">Convolutional Neural Network (CNN)</span> A type of neural network designed to analyze images by looking for patterns in small sections of the image.
    
        <span class="inline-badge">Cross-Validation</span> A method to test how well a model works by training it on different parts of the data and testing on the remaining parts.
    
        <span class="inline-badge">Data Augmentation</span> Techniques to create more training data by making small changes to existing data (like flipping images or adjusting brightness).
    
        <span class="inline-badge">Deep Learning</span> A type of artificial intelligence that uses computer networks to learn patterns from data, similar to how the human brain learns.
    
        <span class="inline-badge">Delta Waves</span> Slow brain waves (0.5-4 Hz) that occur during deep sleep and are important for restorative sleep.
    
        <span class="inline-badge">Early Stopping</span> A technique that stops training when the model stops improving, preventing overfitting.
    
        <span class="inline-badge">EEG (Electroencephalography)</span> A test that records electrical activity in the brain using small sensors placed on the scalp. It's painless and non-invasive.
    
        <span class="inline-badge">FFT (Fast Fourier Transform)</span> A mathematical technique that converts time-based signals into frequency-based information, used to create spectrograms.
    
        <span class="inline-badge">Frequency Bands</span> Different ranges of brain wave frequencies that correspond to different mental states (delta, theta, alpha, beta).
    
        <span class="inline-badge">Hypopnea</span> A partial reduction in breathing during sleep that lasts at least 10 seconds and causes a drop in oxygen levels.
    
        <span class="inline-badge">ImageNet</span> An extensive database of images used to pre-train computer vision models, which helps them learn general image recognition skills.
    
        <span class="inline-badge">Learning Rate</span> How quickly a neural network learns from data. Too fast can cause instability, and too slow can take too long to train.
    
        <span class="inline-badge">MAE (Mean Absolute Error)</span> A measure of prediction accuracy that shows the average difference between predicted and actual values.
    
        <span class="inline-badge">Neural Network</span> A computer system designed to work like the human brain, made up of connected nodes that process information.
    
        <span class="inline-badge">Obstructive Sleep Apnea</span> The most common type of sleep apnea, caused by the throat muscles relaxing and blocking the airway.
    
        <span class="inline-badge">Overfitting</span> When a model learns the training data too well but fails to generalize to new, unseen data.
    
        <span class="inline-badge">Pearson Correlation</span> A measure of how well two variables (like predicted and actual AHI) move together, ranging from -1 to +1.
    
        <span class="inline-badge">Polysomnography</span> A comprehensive sleep study that records multiple body functions during sleep, including brain activity, breathing, and heart rate.
    
        <span class="inline-badge">Regression</span> A type of prediction that estimates a continuous number (like AHI) rather than just categories (like mild/moderate/severe).
    
        <span class="inline-badge">ResNet-18</span> A specific type of neural network with 18 layers that's good at analyzing images and finding patterns in them.
    
        <span class="inline-badge">Residual Connections</span> Shortcuts in neural networks that help information flow better and make training more stable.
    
        <span class="inline-badge">RMSE (Root Mean Square Error)</span> A measure of prediction accuracy that penalizes larger errors more heavily than smaller ones.
    
        <span class="inline-badge">SHHS-1 (Sleep Heart Health Study)</span> An extensive research study that collected sleep data from thousands of participants, which was used as the dataset for this project.
    
        <span class="inline-badge">Sleep Apnea</span> A sleep disorder where breathing repeatedly stops and starts during sleep, often causing loud snoring and daytime tiredness.
    
        <span class="inline-badge">Sleep Stages</span> Different phases of sleep (light sleep, deep sleep, REM sleep) that have distinct brain wave patterns.
    
        <span class="inline-badge">Spectrogram</span> A visual representation of sound or brain signals that shows how different frequencies change over time. It's like a "picture" of the signal.
    
        <span class="inline-badge">STFT (Short-Time Fourier Transform)</span> A mathematical technique used to create spectrograms by analyzing how frequencies change over short time windows.
    
        <span class="inline-badge">Subject-Stratified</span> A way of organizing data so that all data from the same person stays together in either training or testing, preventing data leakage.
    
        <span class="inline-badge">Theta Waves</span> Brain waves (4-8 Hz) that occur during light sleep and drowsiness.
    
        <span class="inline-badge">Transfer Learning</span> A technique where a computer model that's already learned from one type of data (like general images) is adapted to work with new data (like brain signal images).
    
        <span class="inline-badge">Validation</span> The process of testing how well a computer model works on data it hasn't seen before, to make sure it can make accurate predictions in real situations.
    
        <span class="inline-badge">Window</span> A short segment of time (like 30 seconds) used to analyze brain signals, allowing the computer to focus on specific patterns.
        """, unsafe_allow_html=True)

def main():
    st.title("ResNet-18-Based EEG AHI Regression Pipeline")

    st.markdown("""
    <style>
    .inline-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: 500;
        background-color: #ff4b4b;
        color: white;
    }
    h1 {
        text-align: center;
    }
    div[role="tablist"] {
        display: flex;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Abstract", 
        "Introduction", 
        "Literature Review", 
        "Materials & Methods", 
        "Results & Discussion", 
        "Conclusion", 
        "References",
        "Key Terms"
    ])

    with tab1:
        show_abstract()
    with tab2:
        show_introduction()
    with tab3:
        show_literature_review()
    with tab4:
        show_methods()
    with tab5:
        show_results()
    with tab6:
        show_conclusion()
    with tab7:
        show_references()
    with tab8:
        show_terminology()

if __name__ == "__main__":
    main()