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
        <span class="inline-badge">Sleep apnea</span> is a prevalent sleep disorder affecting millions worldwide, with significant cardiovascular and cognitive consequences. We present a <span class="inline-badge">deep learning pipeline</span> for automated <span class="inline-badge">Apnea-Hypopnea Index (AHI)</span> estimation from single-channel EEG recordings using spectrogram-based analysis.
    
        Our approach utilizes the <span class="inline-badge">Sleep Heart Health Study (SHHS-1)</span> dataset, processing C3/A2 EEG signals into 30-second spectrogram windows with comprehensive quality control. We implement a <span class="inline-badge">ResNet-18 architecture</span> with transfer learning from ImageNet pretraining, optimized for continuous AHI regression rather than categorical classification. The model incorporates subject-stratified cross-validation, learning rate warmup, early stopping, and data augmentation to ensure robust generalization.
    
        Results demonstrate moderate performance with a <span class="inline-badge">Root Mean Square Error (RMSE)</span> of 6.8 events/hour and <span class="inline-badge">Pearson correlation</span> of 0.76 on held-out validation data. While systematic biases exist (overprediction at low AHI, underprediction at high AHI), the model provides clinically relevant severity ranking suitable for screening applications. Our work establishes the feasibility of automated sleep apnea assessment from standard EEG recordings.
    
        **Keywords:** Sleep apnea, EEG spectrograms, deep learning, ResNet-18, AHI regression
        """, unsafe_allow_html=True)

def show_introduction():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Introduction")
        st.markdown("""
        <span class="inline-badge">Sleep apnea</span> is a common <span class="inline-badge">sleep disorder</span> in which the upper airway repeatedly narrows or collapses 
    during sleep, causing pauses in breathing (apneas) or shallow breathing episodes (hypopneas). These 
    interruptions fragment sleep and reduce blood oxygen levels, leading to daytime fatigue, morning headaches, 
    impaired concentration, and an increased risk of cardiovascular and metabolic disorders. The severity of 
    sleep apnea is quantified by the <span class="inline-badge">Apnea-Hypopnea Index (AHI)</span>, which measures the number of breathing events per hour 
    of sleep. Accurate AHI assessment is crucial for clinical decision-making.
    
    The high prevalence of sleep apnea underscores the need for more accessible diagnostic tools. In Canada, 
    an estimated 6.4% of adults received a professional diagnosis of sleep apnea in 2016-2017, while 
    population-based assessments suggest that nearly 28.1% of Canadians aged 45 to 85 years have moderate 
    to severe obstructive sleep apnea (AHI ≥15 events/hour) based on STOP-BANG screening (Statistics Canada, 2018; 
    Canadian Longitudinal Study on Aging Team, 2024).
    
    <span class="inline-badge">Electroencephalography (EEG)</span> offers a promising solution for automated sleep apnea assessment. As a non-invasive 
    neuroimaging technique that records electrical activity in the brain through electrodes placed on the scalp, 
    EEG captures the synchronized firing of neurons, producing characteristic waveforms that vary with different 
    states of consciousness and neurological conditions. During sleep, EEG recordings reveal distinct patterns 
    associated with different sleep stages and can detect disruptions caused by sleep disorders such as apnea. 
    The temporal resolution of EEG (milliseconds) makes it ideal for capturing rapid changes in brain activity 
    that occur during sleep-wake transitions and respiratory events, providing a rich source of information for 
    automated analysis.
    
    However, raw EEG signals present significant challenges for automated analysis. They are highly sensitive 
    to artifacts (muscle activity, eye movements, electrode noise), require extensive preprocessing, and contain 
    complex temporal dependencies that are difficult for deep learning models to learn directly. <span class="inline-badge">Spectrograms</span> address 
    these limitations by transforming the time-domain signal into a time-frequency representation that preserves 
    both temporal and spectral information. This transformation makes sleep apnea-related patterns more visually 
    apparent and computationally tractable, effectively highlighting frequency bands associated with different 
    sleep stages (delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-13 Hz, beta: 13-30 Hz) and revealing disruptions in 
    these patterns caused by apnea events. Additionally, spectrograms are more robust to noise and artifacts, 
    as the frequency domain representation naturally filters out many types of interference.
    
    To leverage the advantages of spectrogram-based analysis, we developed a <span class="inline-badge">deep learning pipeline</span> using the 
    <span class="inline-badge">ResNet-18 architecture</span>. This choice was motivated by several factors: its proven performance on image 
    classification tasks makes it well-suited for spectrogram analysis, as spectrograms can be treated as 2D 
    images where spatial patterns correspond to time-frequency relationships. The residual connections in 
    ResNet-18 help maintain gradient flow during training, enabling effective learning of complex hierarchical 
    features from the spectrogram data. ResNet-18's moderate depth (18 layers) provides sufficient 
    representational capacity without overfitting on our limited medical dataset, while the availability of 
    ImageNet pretrained weights enables effective transfer learning, allowing the model to leverage general 
    image recognition capabilities while fine-tuning for the specific task of AHI regression. Finally, 
    ResNet-18's computational efficiency makes it suitable for real-time clinical applications, where rapid 
    inference is essential.
    
    Our approach performs <span class="inline-badge">continuous AHI regression</span> directly from windowed EEG spectrogram images, rather than 
    categorical classification, providing more precise severity estimates that are clinically valuable. The 
    pipeline incorporates subject-stratified training and validation splits to prevent data leakage, 
    comprehensive quality control to ensure reliable spectrogram generation, learning-rate warmup and early 
    stopping to optimize training, and data augmentation to improve generalization. Window-level predictions 
    are aggregated at the subject level to yield per-individual AHI estimates suitable for clinical screening 
    and research applications.
        """, unsafe_allow_html=True)

def show_literature_review():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Literature Review")
        st.markdown("""
        Building on the need for scalable, automated estimation of sleep apnea severity from EEG spectrograms, recent work in automated sleep analysis can be categorized into three complementary areas: <span class="inline-badge">sleep stage scoring</span>, <span class="inline-badge">disorder classification</span>, and <span class="inline-badge">integrated multi-task frameworks</span>.

        <span class="inline-badge">Sleep stage scoring with spectrogram-based deep learning</span>. Spectrograms of EEG preserve both time and frequency information, letting convolutional networks learn patterns that distinguish sleep stages. Li et al. (2022) introduced EEGSNet, a hybrid CNN-BiLSTM model that achieved over 94% accuracy on the Sleep-EDF-8 dataset and strong agreement (κ > 0.77) on several public cohorts by combining learned spectral features with temporal context. Likewise, Tsinalis et al. (2016) showed that an end-to-end CNN trained on single-channel EEG spectrograms can reach balanced F1-scores of approximately 0.81 without manual feature design, demonstrating that deep filters naturally capture stage-specific rhythms.

        <span class="inline-badge">Disorder detection from physiological signals</span>. Beyond staging, identifying specific sleep pathologies enables targeted care. Zhuang and Ibrahim (2022) developed DL-R, a multi-channel CNN that leverages raw EEG, EMG, ECG, and EOG to classify eight sleep disorders with over 95% sensitivity and specificity. Gawhale et al. (2023) extracted deep features from EEG spectrograms and fed them to an ensemble classifier, achieving an overall accuracy of 96.8%, which highlights the feasibility of real-time disorder screening on lightweight devices.

        <span class="inline-badge">Ensemble, multimodal, and multi-task strategies</span>. To combat data imbalance and leverage diverse signals, ensemble and multi-task methods have emerged. Monowar et al. (2025) combined multiple learners with SMOTE-based augmentation to achieve 99.5% cross-validated accuracy for disorder detection, outperforming standalone models. Khanmohmmadi et al. (2025) proposed a multi-task CNN optimized via genetic and Q-learning that jointly predicts sleep deprivation and disorder labels, achieving 98% accuracy by sharing intermediate representations. Cheng et al. (2023) fused parallel CNNs on EEG, ECG, and EMG to perform simultaneous sleep stage and disorder classification, reporting 94.3% staging accuracy and 99.1% disorder accuracy—underscoring the benefit of integrated, multimodal modelling.

        While these classification advances are impressive, most focus on categorical labels rather than continuous severity measures. Our ResNet-18-based pipeline extends this work by directly predicting per-subject AHI from windowed EEG spectrograms. By using subject-stratified splits to avoid leakage, monitoring class balance, employing learning-rate warm-up, data augmentation, and early stopping, and aggregating predictions at the individual level, we aim to deliver an end-to-end method for continuous estimation of sleep apnea severity.
        """, unsafe_allow_html=True)
        st.subheader("Reproduction and Evaluation of Tanci & Hekim (2025) Findings")
        st.markdown("""
        Tanci and Hekim (2025) developed a four-class sleep apnea classifier by transforming 30-second windows of EEG signals into STFT spectrograms and feeding them into a YOLOv8 network. Each spectrogram corresponds to one of four AHI-based categories: healthy (AHI < 5), mild (5-14.9), moderate (15-29.9), or severe (≥30) events/hour. While a naïve baseline assumes 25% accuracy per class, YOLOv8 achieved a total correct classification (TCC) of 93.7%, outperforming both ResNet64 (93.0%) and YOLOv5 (88.2%). The authors conclude that YOLOv8 offers both rapid inference and high accuracy for EEG-based sleep apnea staging.

        The dataset used in this study is C3-A2 channel EEG signals from PSG recordings in the Physionet database (PhysioBank ATM), comprising data from 25 subjects with varying degrees of sleep apnea severity. The C3-A2 electrode placement offers a clinically standard, high-quality window into the brain's response to breathing disruptions, while keeping preprocessing and model complexity manageable. This single-channel approach captures the essential neural signatures associated with sleep apnea events without the computational overhead of multi-channel analysis, making it suitable for both research and potential clinical deployment.

        **Spectrogram Generation Methodology**

        Tanci and Hekim (2025) provide a detailed mathematical foundation for their spectrogram generation approach. The spectrogram represents the frequencies where the energy of the signal is maximum, effectively transforming the two-dimensional EEG waveform into a three-dimensional time-frequency representation. In other words, a spectrogram shows the energy change of a signal over time, defined as the power distribution of the Short-Time Fourier Transform (STFT).

        For the STFT application, a moving window function g(t−τ) is applied to the signal x(t) at time τ. Each window is moved by τ in the time domain, and these changes in the time interval are displayed in the windows. The STFT is mathematically defined as:

        **X(τ,f) = ∫ from -∞ to ∞ of x(t)g(t-τ) * exp(-j2πft) dt**

        Where:  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**x(t)**: Analyzed signal (EEG waveform)  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**g(t−τ)**: Windowing function (Hann window in their implementation)  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**τ**: Time-shifting parameter  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**f**: Frequency parameter  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**exp(-j2πft)**: Complex exponential function used in the Fourier transform

        This integral allows the signal to be analyzed over a given time interval (τ) and at a given frequency (f), enabling simultaneous analysis of both time and frequency information. When g(t−τ) is considered a windowing function, the STFT analyzes both the time and frequency information of a signal simultaneously, providing the foundation for spectrogram generation.

        Their approach follows a comprehensive deep learning classification pipeline, as illustrated in their article, which transforms raw EEG signals through multiple processing stages to produce categorical sleep apnea severity classifications.

        *Figure: Tanci & Hekim's (2025) deep learning classification pipeline for sleep apnea severity detection. The process begins with raw EEG signals, which are segmented into 30-second windows, converted to spectrograms for time-frequency analysis, processed through deep learning models, and finally classified into severity categories (normal, mild, moderate, severe).*
        """, unsafe_allow_html=True)
        st.image("assets/tanci_hekim_pipeline.png", use_container_width=True)
        st.markdown("""
        *Figure: Tanci & Hekim's (2025) deep learning classification pipeline for sleep apnea severity detection. The process begins with raw EEG signals, which are segmented into 30-second windows, converted to spectrograms for time-frequency analysis, processed through deep learning models, and finally classified into severity categories (normal, mild, moderate, severe).*

        **Visual Examples of EEG Signals and Spectrograms**

        To illustrate the relationship between raw EEG signals and their spectrogram representations, Tanci and Hekim (2025) provide visual examples for each sleep apnea severity class. These examples demonstrate how different patterns in the time domain correspond to distinct frequency-domain characteristics.
        """, unsafe_allow_html=True)
        st.image("assets/tanci_hekim_eeg_spectrograms.png", use_container_width=True)
        st.markdown("""
        *Figure: EEG signals and spectrogram examples for different sleep apnea severity levels: (A) Mild, (B) Moderate, (C) Severe, and (D) Healthy. Each panel displays a raw EEG signal (left) and its corresponding time-frequency representation (spectrogram, right), illustrating the visual patterns associated with varying degrees of sleep apnea. The spectrograms are from 30-second time slices containing representative samples from each class, enabling clear visual comparison of patterns between severity categories.*

        We recreated Tanci and Hekim's (2025) exact spectrogram setup in simple steps: we cut the C3A2 EEG into 30-second chunks, ran a Hann-window STFT with a 256-point FFT and 50% overlap, converted the results to decibels, and skipped any chunk that had empty (all-floor) columns. We saved these spectrograms as 150 dpi viridis PNGs and then ran them through each of their published models—ResNet64, YOLOv5, and YOLOv8—to classify sleep apnea severity. Finally, we evaluated each model on a balanced hold-out test set of 400 samples (100 per class) to compare their performance.

        <span class="inline-badge">ResNet64 Performance Analysis</span>. **ResNet64** achieved a balanced accuracy of 85.50%, with macro-averaged precision of 0.8703, recall of 0.8550, and F1-score of 0.8555. The model demonstrated strong discriminative capabilities across all classes, as evidenced by high AUC values ranging from 0.95 to 0.99 and Average Precision values from 0.90 to 0.97.

        **Per-Class Performance Breakdown:**

        **Severe Apnea** (Best Performance): The model achieved exceptional performance for severe cases with precision = 0.91, recall = 0.92, and F1-score = 0.92. The confusion matrix reveals that 92 out of 100 true severe cases were correctly classified (92.0% accuracy), with only 7% misclassified as mild and 1% as moderate. The ROC curve shows an AUC of 0.99, and the Precision-Recall curve demonstrates an Average Precision of 0.97, indicating near-perfect discriminative power for this critical severity level.

        **Normal Cases** (High Precision): The normal class showed excellent performance with perfect precision (1.00), meaning no other classes were misclassified as normal. However, the recall was 75% (75 out of 100 true normal cases correctly identified), with 9% misclassified as mild, 14% as moderate, and 2% as severe. The ROC curve achieved an AUC of 0.99, and the Precision-Recall curve shows an Average Precision of 0.97, highlighting the model's strong ability to avoid false positives for normal cases.

        **Moderate Apnea** (Moderate Performance): Moderate cases showed balanced performance with precision = 0.82, recall = 0.81, and F1-score = 0.81. The confusion matrix indicates 81 out of 100 true moderate cases were correctly classified, with 15% misclassified as mild and 4% as severe. The ROC curve shows an AUC of 0.95, and the Precision-Recall curve demonstrates an Average Precision of 0.90, reflecting the model's solid but slightly lower discriminative power for this intermediate severity level.

        **Mild Apnea** (Weakest Performance): Mild cases showed the most challenging performance with precision = 0.75, recall = 0.94, and F1-score = 0.84. While the model correctly identified 94 out of 100 true mild cases (high recall), it suffered from false positives where 15 moderate and 9 normal cases were incorrectly predicted as mild. The ROC curve shows an AUC of 0.97, and the Precision-Recall curve demonstrates an Average Precision of 0.92, indicating good discriminative power but challenges in maintaining high precision.

        **Overall Assessment:** ResNet64 demonstrates robust performance across all sleep apnea severity classes, with particularly strong performance for severe and normal cases. The model's ability to achieve high AUC values (0.95-0.99) across all classes indicates excellent discriminative capabilities, while the varying precision-recall trade-offs reflect the inherent challenges in distinguishing between adjacent severity categories, particularly between mild and moderate cases.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/ResNet64_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/ResNet64_roc_curves_balanced.png")
        with col3:
            st.image("assets/ResNet64_pr_curves_balanced.png")
        st.markdown("""
        <span class="inline-badge">YOLOv5 Performance Analysis</span>. **YOLOv5** outperformed ResNet64 with balanced accuracy of 88.75%, macro precision of 0.8972, recall of 0.8875, and F1 of 0.8887. The model demonstrated exceptional discriminative capabilities across all classes, as evidenced by high AUC values ranging from 0.98 to a perfect 1.00 and robust Average Precision values from 0.95 to 0.99.

        **Per-Class Performance Breakdown:**

        **Normal Cases** (Best Performance): YOLOv5 achieved near-perfect performance for normal cases with precision = 0.99, recall = 0.85, and F1-score = 0.91. The confusion matrix reveals that 85 out of 100 true normal cases were correctly classified, with only 9% misclassified as mild, 4% as moderate, and 2% as severe. The ROC curve shows a perfect AUC of 1.00, and the Precision-Recall curve demonstrates an Average Precision of 0.99, indicating exceptional discriminative power and minimal false positives for normal cases.

        **Severe Apnea** (Strong Performance): The model showed robust performance for severe cases with precision = 0.94, recall = 0.88, and F1-score = 0.91. The confusion matrix indicates 88 out of 100 true severe cases were correctly classified, with only 9% misclassified as mild and 3% as moderate. The ROC curve shows an AUC of 0.99, and the Precision-Recall curve demonstrates an Average Precision of 0.98, highlighting the model's strong ability to identify severe cases with high confidence.

        **Mild Apnea** (High Recall): Mild cases showed the highest recall (0.96) among all classes, with precision = 0.79 and F1-score = 0.86. The confusion matrix reveals that 96 out of 100 true mild cases were correctly identified, demonstrating excellent sensitivity. However, the model suffered from false positives where 26 cases (10 moderate, 9 normal, 7 severe) were incorrectly predicted as mild. The ROC curve shows an AUC of 0.98, and the Precision-Recall curve demonstrates an Average Precision of 0.95, indicating good discriminative power but challenges in maintaining high precision.

        **Moderate Apnea** (Balanced Performance): Moderate cases showed balanced performance with precision = 0.88, recall = 0.86, and F1-score = 0.87. The confusion matrix indicates 86 out of 100 true moderate cases were correctly classified, with 10% misclassified as mild and 4% as severe. Additionally, 12 cases were falsely predicted as moderate when they belonged to other classes. The ROC curve shows an AUC of 0.98, and the Precision-Recall curve demonstrates an Average Precision of 0.96, reflecting solid discriminative power for this intermediate severity level.

        **Overall Assessment:** YOLOv5 demonstrates superior performance compared to ResNet64, achieving higher balanced accuracy (88.75% vs 85.50%) and more consistent performance across all classes. The model's ability to achieve perfect AUC (1.00) for normal cases and high AUC values (0.98-0.99) across all classes indicates excellent discriminative capabilities. While mild cases show the highest recall, the precision-recall trade-offs reflect the inherent challenges in distinguishing between adjacent severity categories, particularly the tendency to over-classify mild cases.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/YOLOv5_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/YOLOv5_roc_curves_balanced.png")
        with col3:
            st.image("assets/YOLOv5_pr_curves_balanced.png")
        st.markdown("""
        <span class="inline-badge">YOLOv8 Performance Analysis</span>. **YOLOv8**, contrary to the original report, yielded a lower balanced accuracy of 72.25%, with macro precision of 0.7409, recall of 0.7225, and F1 score of 0.7247. The model demonstrated moderate discriminative capabilities across all classes, as evidenced by AUC values ranging from 0.89 to 0.95 and Average Precision values from 0.78 to 0.91, significantly underperforming compared to both ResNet64 and YOLOv5.

        **Per-Class Performance Breakdown:**

        **Normal Cases** (Best Precision): YOLOv8 achieved the highest precision for normal cases with precision = 0.93, recall = 0.66, and F1-score = 0.77. The confusion matrix reveals that 66 out of 100 true normal cases were correctly classified, with 16% misclassified as mild and 12% as moderate. The ROC curve shows an AUC of 0.95, and the Precision-Recall curve demonstrates an Average Precision of 0.91, indicating good discriminative power for normal cases despite the lower recall.

        **Severe Apnea** (Best Recall): The model showed the highest recall for severe cases with precision = 0.72, recall = 0.80, and F1-score = 0.76. The confusion matrix indicates 80 out of 100 true severe cases were correctly classified, with 12% misclassified as mild and 8% as moderate. Notably, no severe cases were misclassified as normal. The ROC curve shows an AUC of 0.91, and the Precision-Recall curve demonstrates an Average Precision of 0.83, highlighting the model's ability to identify most severe cases.

        **Moderate Apnea** (Weak Performance): Moderate cases showed poor performance with precision = 0.68, recall = 0.71, and F1-score = 0.69. The confusion matrix reveals that 71 out of 100 true moderate cases were correctly classified, with 13% misclassified as mild and 13% as severe. Additionally, 13 cases were falsely predicted as moderate when they belonged to other classes. The ROC curve shows an AUC of 0.90, and the Precision-Recall curve demonstrates an Average Precision of 0.78, reflecting the model's challenges in distinguishing moderate cases from adjacent categories.

        **Mild Apnea** (Weakest Performance): Mild cases showed the poorest performance with precision = 0.64, recall = 0.72, and F1-score = 0.68. The confusion matrix indicates 72 out of 100 true mild cases were correctly classified, with 14% misclassified as moderate and 12% as severe. The model also suffered from false positives where 28 cases were incorrectly predicted as mild. The ROC curve shows the lowest AUC of 0.89, and the Precision-Recall curve demonstrates an Average Precision of 0.81, indicating the weakest discriminative power among all classes.

        **Overall Assessment:** YOLOv8 demonstrates significantly inferior performance compared to both ResNet64 and YOLOv5, achieving the lowest balanced accuracy (72.25% vs 85.50% and 88.75%) and showing substantial challenges in distinguishing between adjacent severity categories. The model's AUC values (0.89-0.95) and Average Precision values (0.78-0.91) are notably lower than the other models, particularly for mild and moderate cases. The high inter-class confusion, especially between mild and moderate categories, suggests fundamental limitations in the model's ability to learn discriminative features for sleep apnea severity classification.
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("assets/YOLOv8_confusion_matrix_balanced.png")
        with col2:
            st.image("assets/YOLOv8_roc_curves_balanced.png")
        with col3:
            st.image("assets/YOLOv8_pr_curves_balanced.png")
        st.markdown("""
        Tanci and Hekim's (2025) study presents an innovative application of EEG spectrograms and the YOLOv8 architecture for sleep apnea classification; however, several aspects of their design raise concerns. By labelling every 30-second spectrogram according to the subject's overall AHI category—even when many windows may not contain apnea events—the model is trained on noisy and potentially misleading targets. Their random division of spectrogram images into training and test sets allows data from the same recording to appear in both, which can artificially boost performance by allowing the network to memorize subject-specific signal patterns rather than learn generalizable features. Moreover, the decision to train each backbone for a fixed number of epochs without any learning-rate scheduling or early stopping risks both under- and over-training, since there is no mechanism to detect when the model has truly converged. The inclusion of spectrograms that contain little or no meaningful signal—due to the absence of any quality‐control filtering—further dilutes the training data and can hinder model convergence. Finally, framing AHI estimation as a four-class classification task overlooks the continuous nature of apnea severity, potentially obscuring critical clinical nuances near category thresholds. Together, these methodological choices suggest that the reported 93.7% total correct classification may overstate the model's actual ability to generalize to unseen patients and to produce clinically useful estimates of sleep apnea severity.
        """, unsafe_allow_html=True)

        st.markdown("""
        In contrast to Tanci and Hekim's classification approach, our research implements a comprehensive regression pipeline for continuous AHI estimation. Our methodology begins with <span class="inline-badge">raw EEG signals</span> from the C3/A2 channel of the SHHS-1 dataset, comprising 100 subjects with varying degrees of sleep apnea severity. The <span class="inline-badge">signal segmentation</span> process divides these recordings into 30-second windows with 50% overlap, following the AASM standard for sleep analysis. We implement comprehensive <span class="inline-badge">quality control</span> through variance filtering (standard deviation > 5.0 and < 200.0) and artifact rejection to ensure only high-quality epochs are retained for analysis.

        The <span class="inline-badge">spectrogram generation</span> employs Short-Time Fourier Transform (STFT) with 256-point FFT and Hann windowing, followed by global normalization to ensure consistent color mapping across all subjects. The resulting <span class="inline-badge">image processing</span> produces 224×224 pixel PNG spectrograms using the viridis colormap, optimized for frequency visualization and deep learning model input.

        Our <span class="inline-badge">deep learning architecture</span> utilizes a ResNet-18 backbone with ImageNet pretrained weights, enabling effective transfer learning for the limited medical dataset. We replace the final classification layer with a custom regression head that outputs a single continuous AHI value, allowing for precise severity estimation rather than categorical classification. The <span class="inline-badge">training strategy</span> employs subject-stratified 5-fold cross-validation to prevent data leakage between individuals, ensuring realistic performance estimates on unseen subjects. We implement several <span class="inline-badge">optimization techniques</span> including learning rate warmup, early stopping, and data augmentation to improve model generalization and prevent overfitting.

        The <span class="inline-badge">prediction and evaluation</span> process operates at two levels: individual 30-second window predictions are generated first, then aggregated at the subject level by averaging to yield per-individual AHI estimates. This approach provides continuous AHI regression output rather than categorical classification, offering more clinically relevant precision for severity assessment. Performance is evaluated using standard regression metrics including Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Pearson correlation coefficient.

        Our approach offers several key advantages over classification-based methods. The <span class="inline-badge">data leakage prevention</span> through subject-stratified splits ensures realistic validation performance that generalizes to unseen individuals. The <span class="inline-badge">clinical relevance</span> of continuous AHI estimates provides precise severity assessment suitable for treatment planning and monitoring. Comprehensive <span class="inline-badge">quality assurance</span> measures ensure reliable spectrogram generation and model training. Finally, the <span class="inline-badge">scalable design</span> using single-channel EEG makes the pipeline suitable for clinical deployment in various healthcare settings.
        """, unsafe_allow_html=True)
        st.image("assets/ahi_regression_pipeline.png", use_container_width=True)
        st.markdown("""
        *Figure: Overview of the ResNet-18-based EEG AHI regression pipeline. The process begins with raw EEG signals, which are segmented, transformed into spectrograms, processed by a deep learning model, and yield continuous AHI predictions.*
        """, unsafe_allow_html=True)

def show_methods():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.header("Materials & Methods")
        st.subheader("Dataset")
        st.markdown("""
        We utilize the <span class="inline-badge">Sleep Heart Health Study 1 (SHHS-1)</span> dataset, the baseline examination cycle of the landmark multi-center cohort study implemented by the National Heart Lung & Blood Institute. SHHS-1 was conducted between November 1995 and January 1998, with polysomnography recordings obtained from 6,441 participants recruited from existing epidemiological studies.
    
        <span class="inline-badge">Participant recruitment</span> was conducted from nine existing epidemiological studies including the Framingham Offspring Cohort, Atherosclerosis Risk in Communities (ARIC) study sites in Hagerstown and Minneapolis/St. Paul, Cardiovascular Health Study (CHS) sites in Hagerstown, Sacramento and Pittsburgh, and Strong Heart Study sites in South Dakota, Oklahoma, and Arizona. Inclusion criteria required participants to be 40 years or older with no history of sleep apnea treatment, tracheostomy, or current home oxygen therapy. Several cohorts over-sampled snorers to increase the study-wide prevalence of sleep-disordered breathing.
    
        <span class="inline-badge">SHHS-1 polysomnography recordings</span> were obtained in unattended settings, typically in participants' homes, by trained and certified technicians. The recording montage included C3/A2 and C4/A1 EEG channels sampled at 125 Hz, bilateral electrooculograms (EOG) at 50 Hz, submental electromyogram (EMG) at 125 Hz, thoracic and abdominal excursions via inductive plethysmography at 10 Hz, nasal-oral thermocouple airflow detection at 10 Hz, finger-tip pulse oximetry at 1 Hz, and ECG at 125 Hz. Additional sensors monitored body position and ambient light levels.
    
        <span class="inline-badge">Data subset selection</span> involved stratified sampling of 100 subjects from the available 6,441 participants to ensure balanced representation across apnea severity categories. The final dataset includes 49 healthy subjects (AHI < 5), 30 mild cases (AHI 5-14.9), 14 moderate cases (AHI 15-29.9), and 7 severe cases (AHI ≥ 30), providing a clinically relevant distribution for model development and validation.
    
        <span class="inline-badge">EEG signal processing</span> focuses on the C3/A2 EEG channel, which is automatically detected from available channels using preference matching for "EEG(sec)", "EEG2", "EEG 2", or "EEG sec" labels. This channel provides the primary input for spectrogram generation, with signals processed at their native 125 Hz sampling rate to preserve temporal and frequency resolution.
    
        The SHHS-1 dataset represents the largest and most comprehensive baseline sleep study, providing essential data for investigating the cardiovascular consequences of sleep-disordered breathing. The comprehensive physiological monitoring, large sample size, and standardized protocols make SHHS-1 ideal for developing and validating automated sleep apnea severity estimation methods that can generalize to diverse populations and clinical settings.
    
        <span class="inline-badge">AHI Distribution Analysis</span>. The dataset exhibits a characteristic right-skewed distribution of AHI values, with the majority of subjects having low to moderate apnea severity. This distribution reflects the natural prevalence of sleep apnea in the general population, where most individuals have minimal or mild symptoms, while severe cases are less common but clinically significant.
        """, unsafe_allow_html=True)
        st.image("assets/ahi_distribution_histogram.png", use_container_width=True)
        st.markdown("""
        <span class="inline-badge">Data Distribution Characteristics</span>. The histogram reveals a heavily right-skewed distribution with peaks at low AHI values (0-1 and 5-6), indicating that most subjects have minimal or mild sleep apnea. The validation set maintains similar statistical properties to the training set, ensuring representative model evaluation. This distribution highlights the challenge of predicting rare severe cases while maintaining accuracy across the full severity spectrum.
        """, unsafe_allow_html=True)
        st.subheader("Spectrogram Generation & Quality Control")
        st.markdown("""
        We transform raw EEG signals into <span class="inline-badge">spectrogram images</span> using a comprehensive pipeline designed for clinical applications. The process begins with <span class="inline-badge">signal segmentation</span> into 30-second windows (AASM standard) with 50% overlap to capture breathing event transitions, followed by <span class="inline-badge">STFT computation</span> using 256-point FFT with 128-point overlap for smooth spectrograms.
    
        <span class="inline-badge">Global contrast normalization</span> is computed across all subjects to ensure consistent color mapping (vmin=-53.68 dB, vmax=32.78 dB). <span class="inline-badge">Quality control</span> is implemented through variance filtering to remove flat epochs (std ≤ 5.0) indicating sensor issues and artifact rejection for noisy epochs (std ≥ 200.0) with excessive variance.
    
        The <span class="inline-badge">image generation</span> produces 224×224 pixel spectrograms using the viridis colormap for optimal frequency visualization, exported in PNG format with tight bounding boxes and no padding. See examples below.
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
        The output is organized in per-subject directories (`resnet224_color/{subject_id}/`) with systematic naming convention `{subject_id}_{window_index:04d}.png` and comprehensive metadata tracking including paths, AHI values, and quality metrics. Each subject typically generates 1,500-2,200 spectrogram windows, with quality control filtering ensuring only high-quality epochs are retained.
    
        <span class="inline-badge">Cross-validation strategy</span> employs 5-fold stratified group cross-validation to prevent data leakage between subjects. Each fold contains approximately 31,000-36,000 spectrogram windows, with subjects randomly assigned to folds while maintaining balanced representation of apnea severity categories. This approach ensures that predictions on unseen subjects provide realistic estimates of model generalization performance.
    
        <span class="inline-badge">Data organization</span> includes comprehensive metadata tracking with window-level information stored in `window_metadata.csv` and fold assignments in `spectrogram_splits.csv`. The dataset contains over 166,000 high-quality spectrogram windows across 100 subjects, providing sufficient data for robust model training while maintaining clinical relevance through stratified sampling.
    
        This pipeline preserves both temporal and frequency information while creating standardized inputs suitable for deep learning model training and evaluation.
        """, unsafe_allow_html=True)
        st.subheader("ResNet-18 Model Architecture & Training")
        st.markdown("""
        Our <span class="inline-badge">ResNet-18 architecture</span> leverages transfer learning from ImageNet pretraining to perform <span class="inline-badge">AHI regression</span> from EEG spectrogram images. The model combines the proven feature extraction capabilities of ResNet-18 with a custom regression head optimized for continuous severity prediction.
    
        <span class="inline-badge">Model Architecture</span>. We utilize the standard ResNet-18 backbone with ImageNet pretrained weights, replacing the final classification layer with a single-output linear regression head. The architecture processes 224×224 RGB spectrogram images through 18 residual layers organized in 4 blocks, with each block containing 2 residual units. The final global average pooling layer reduces spatial dimensions to 512 features, which are then mapped to a single AHI value through the custom fully-connected layer.
    
        <span class="inline-badge">Transfer Learning Strategy</span>. We employ differential learning rates to optimize the pretrained backbone and new regression head separately. The backbone layers use a reduced learning rate (1e-5) to preserve learned features while allowing fine-tuning, while the regression head uses the full learning rate (1e-4) to learn task-specific mappings. This approach balances feature reuse with task adaptation, enabling efficient training on the limited spectrogram dataset.
    
        <span class="inline-badge">Loss Function & Optimization</span>. We use <span class="inline-badge">Smooth L1 Loss (Huber Loss)</span> for regression, which combines the benefits of L1 and L2 losses by being less sensitive to outliers than mean squared error while maintaining smooth gradients. The optimizer is Adam with weight decay regularization, and we implement gradient clipping (norm ≤ 1.0) to prevent exploding gradients during training.
        """, unsafe_allow_html=True)
        st.markdown("""
        <span class="inline-badge">Training Pipeline</span>. The training process incorporates several best practices for robust model development:
    
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="inline-badge">Learning Rate Warmup</span>: Gradual learning rate increase over 5 epochs (0% to 100%) to stabilize early training  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="inline-badge">ReduceLROnPlateau Scheduling</span>: Automatic learning rate reduction by 50% when validation loss plateaus for 5 epochs  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="inline-badge">Early Stopping</span>: Training termination after 10 epochs without validation improvement to prevent overfitting  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="inline-badge">Data Augmentation</span>: Random horizontal flips and color jittering (brightness ±20%, contrast ±20%) to improve generalization  
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="inline-badge">Subject-Level Aggregation</span>: Window predictions averaged per subject for final AHI estimation
    
        <span class="inline-badge">Data Processing</span>. Spectrogram images are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) and resized to 224×224 pixels to match ResNet input requirements. Each 30-second window is labeled with the subject's overall overnight AHI value, enabling the model to learn the relationship between local spectral patterns and global apnea severity.
 
        <span class="inline-badge">Training Configuration</span>. The model is trained for up to 30 epochs with a batch size of 16, using 5-fold subject-stratified cross-validation to prevent data leakage. Training utilizes 80% of subjects (folds 1-4) while validation uses the remaining 20% (fold 0), ensuring realistic performance estimates on unseen individuals. The dataset contains 166,478 high-quality spectrogram windows across 100 subjects, with class-balanced sampling maintained through stratified fold assignment.
    
        <span class="inline-badge">Evaluation Metrics</span>. Model performance is assessed using subject-level mean squared error (MSE) and mean absolute error (MAE) between predicted and true AHI values. Window-level predictions are aggregated per subject by averaging, reflecting the clinical need for per-individual severity estimates. This approach ensures that the model's performance directly corresponds to its utility in clinical decision-making.
    
        The ResNet-18 architecture provides an optimal balance between model complexity and performance, leveraging proven deep learning techniques while maintaining computational efficiency suitable for real-time clinical applications.
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