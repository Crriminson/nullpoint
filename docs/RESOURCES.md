# Resources

## Prior work this builds on or answers
* **Noddle** (head-controlled games; built by someone with nystagmus, for whom eye tracking failed): https://devpost.com/software/noddle
* **BlinkTalk** (eye/blink AAC for ALS, MediaPipe + LightGBM): https://devpost.com/software/blinktalk-0wzjrk
* **Blink-to-Code** (MediaPipe blink → Morse, duration thresholds calibrated per user): https://arxiv.org/abs/2508.09344

## Nystagmus science
* Sarvananthan et al. 2009, *The prevalence of nystagmus: the Leicestershire nystagmus survey* (IOVS): 24.0 per 10,000; neurological nystagmus the largest group: https://pubmed.ncbi.nlm.nih.gov/19458336/
* INS clinical overview, waveforms, null zone, foveation (GeneReviews, NBK3822): https://ncbi.nlm.nih.gov/books/NBK3822
* Infantile nystagmus, an optometrist's perspective (null zone within 10° of primary in 73 %; intensity = amp × freq): https://www.dovepress.com/infantile-nystagmus-an-optometrists-perspective-peer-reviewed-fulltext-article-OPTO
* Dunn, M. J., *An automated segmentation approach to calibrating infantile nystagmus waveforms* (open MATLAB tools, CC BY-SA 4.0): https://reshare.ukdataservice.ac.uk/853460
* Dell'Osso OMLAB eye-movement database (digitised INS recordings, NAFX tools): https://bop.unibe.ch/JEMR/article/view/2312/6720
* Classification of infantile nystagmus waveforms: https://pubmed.ncbi.nlm.nih.gov/27125578/
* StabilEyes: periodic pupil motion from phone video (CMU RI): https://www.ri.cmu.edu/publications/towards-stabileyes-detecting-repetitive-pupil-motion-in-nystagmus-using-discrete-period-quadrature
* Webcam OKN detection via limbus optic flow: https://pubmed.ncbi.nlm.nih.gov/25151522/
* VertiGo, webcam nystagmus detection pilot (59.6 % accuracy, a cautionary baseline): https://ouci.dntb.gov.ua/works/45EkrXO4

## Oscillation cancellation
* Riviere & Thakor, WFLC tremor modelling and cancelling in human-machine interfaces: https://www.ri.cmu.edu/publications/modeling-and-canceling-tremor-in-human-machine-interfaces
* On-line cancelling of pathological tremor for computer input (WFLC beats low-pass): https://ri.cmu.edu/?p=14217

## Tools
* MediaPipe Face Landmarker, Python Tasks API: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
* Model: `face_landmarker.task` (float16) from `storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/`

## Datasets considered
* No public nystagmus **video** dataset found. Plan: synthetic generator + own recordings.
* Blink EEG datasets (voluntary vs involuntary): github.com/meagmohit/BLINK. EEG, not video, so not used.
* mEBAL2 (image eyeblink, spontaneous only, licence request): not needed.

## Prior art and context (added 17 Nov)
* Google Look to Speak (2020), glance-based phrase selection on Android: https://experiments.withgoogle.com/looktospeak
* ConVNG, smartphone video-nystagmography with CNNs (J Neurol 2022): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10129923/
* Rosengren et al. 2020, robust calibration of eye tracking during nystagmus: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7005097/
* Tobii Dynavox, factors that affect gaze interaction (lists nystagmus): https://www.tobiidynavox.com/blogs/support-articles/what-factors-in-your-environment-can-affect-gaze-interaction
* Apple Vision Pro accessibility (pointer control, single-eye input): https://zmknox.com/2024/02/18/vision-accessibility-on-apple-vision-pro
* Vision Pro user with nystagmus (forum report, anecdotal): https://discussions.apple.com/thread/255463705
* Null-zone reading training in infantile nystagmus (2019): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6926938/

