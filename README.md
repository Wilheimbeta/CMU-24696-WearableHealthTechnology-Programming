# CMU-24665-WearableHealthTechnology-Programming
The repository is used for recording the programming for 24696. The group member includes Chester Xiao, Debasmita Kanungo, Owen Liston, Yiman Wu and Zhengyang Bi. 

For detailed instruction, see GUIDE.md.

Link to Completed Programming: https://drive.google.com/file/d/1qr_pexqpEJVYagTLsC5HZ__eJs3K-UM5/view?usp=drive_link

# Project Description

HAR with great generalizability can provide clinicians with real-world data, which assists on more costonmized patient care and low-cost research.

The ideal generalized HAR model is expected to have following three features:
1. Reduction on clinic burden and improvement on rural healthcare access through reduction of cost and enabling remote monitoring by wearable-based HAR.
2. Reduction on limitation of expensive prorietary ecosystems for advanced health tracking through robustness of models on cross-devices scenario. 
4. Transparent evaluation and explicit reporting of effects and parameter trade-offs promotes reproducible and clinically interpretable HAR research for future improvement 

For fulfilling this expectation, LIMU-BERT-X and SSL-Wearable (HARNet) are induced. Here are some features attractive for selection. 


LIMU-BERT-X: Random Crops 
             20 step crops from 120 step windows 
             Timestamp alignment 
             No unit rescaling
             Masked signal modeling

SSL(ResNet): Fixed 10 second windows
             30 Hz Interpolation 
             Gravity Scaling 
             Multi-task augmentation

However, only selection of two models are not enough. There is still a lack of research on human activity recognition model generalizability on large scale, unlabeled datasets. To be specific, human activity recognition models that achieve high accuracies in controlled lab settings, fail to maintain performance in real world scenario. Supervised models overfit to specific populations and sensor frequencies and fail to maintain high accuracy across different model architectures and new populations. Current research lacks a robust framework to leverage unlabeled data to generalize across models. According to background research, a technique "pretraining" would be assistive to generalization of HAR models. 

Based on the demand statement, the project attempts to solve the engineering problem: Does pre-training on large, unlabeled datasets improve accuracy and generalizability across different device types and datasets? 

Based on Engineering problem, there are three aims for research on solution:
1. Establish a baseline for models from scratch and pre-trained on large unlabeled datasets
2. Investigate the relationship between self-supervised weights and model architecture, amount of labeled training data, input data, and more
3. Test generalization of models on data collected in this study in a controlled and uncontrolled real world setting 

For indication of solution, there are three success criteria:
1. Improved Accuracy
Pre-training expected to yield significantly higher accuracy than scratch in out-of-distribution(OOD) experiments.

2. In-the-Wild Robustness
Pre-training expected to have superior performance with limited labeled data compared to models from scratch.

3. Cross-Protocol Stability
Pre-training should maintain robustness across datasets collected via different protocols.

To reach all success criteria, 9 experiments are designed. 

# Experiment Outline
Exp 0: Baseline from scratch and pretrained LOSO testing on all architectures → Aim 1

Result: 
As a baseline (from scratch), all architecture models achieved high accuracy on phone data but performed poorly on watch data.
Self-supervised pre-training enhances performance on the sensor type (watch or phone) for the model’s original sensor type.

Exp 1: Labeled data reduction by fractionating the total data LOSO testing → Aim 2

Result:
Self-supervised pretraining does not reduce the amount of supervised training data needed, both pretrained and scratch models demonstrate the same trend.

Exp 2: LIMU-BERT-X generalizability to watch data with varying pre-training utilizing LOSO testing → Aim 2
Exp 3: SSL-Wearables generalizability to  phone data with varying pre-training utilizing LOSO testing → Aim 2

Result: 
Additional pre-training on a new dataset does not increase accuracy.

Exp 4: 3-channel (accel) vs 6-channel (accel + gyro) sensor comparison utilizing LOSO testing → Aim 2

Result: 
The addition of a gyroscope improves accuracy most prominently for phone data depending on the model’s architecture.

Exp 5&6: Cross-dataset compatibility (train on one dataset, test on another) with various datasets and variables → Aim 1 and 2

Result:
As a baseline, models trained and tested on watch data perform better than phone based models .
Self-supervised pretraining increases cross-dataset generalizability on three activities: standing, sitting, walking.
Self-supervised pre-training increases watch generalizability for 5 activities, whereas phone results are direction dependent.

Exp 7: Labeled data reduction by including and excluding entire subject’s data → Aim  2

Result:
Training on more subjects yields better results and pre-training does not significantly perform better than models from scratch.

Exp 8: Cross dataset and LOSO testing on our collected datasets (controlled and uncontrolled) → Aim 3
Addtional Information:
Data collection protocol consisted of five activities performed in a controlled and uncontrolled manner.

Result:
SSL-Wearables performs best when cross testing on our data with HHAR, however the models perform differently on uncontrolled versus controlled.
Uncontrolled data aligns best with In-The-Wild phone pre-training domain.
Sensor location has an impact on over accuracy as well as accuracy per activity when trained on LIMU-BERT X data.

# Conclusion
From 9 experiments, the main takeaway is that Pre-training domain matters more than architecture.
1. 6ch significantly improves scratch models performance by 15% in accuracy.
2. Domain alignment is a stronger predictor of success than architectural complexity.
3. Significant accuracy drop when moving from benchmarks (73-88%) to real-world data (20-58%).
In conclusion, pre-trained models achieve ~87% accuracy with only 25% of labeled data. Pre-training is proved as a powerful and efficient substitute for exhaustive labeling according to research.

There are also several limitations worth acknowledging for current work:

Data Imbalance & Scope
For baseline dataset, HHAR, contains both phone and watch data. There has no cross-device settings in our collected data (same IMU set)
For dataset collected, it is only partially analyzed.

Population & Generalization
Only 5 activity classes tested, which subjects limited to healthy young adults. Findings may not transfer to clinical or elderly subjects.

Performance Measurement
There has no standardized metric exists to evaluate pretraining quality across devices and datasets. 
Lack of objective ground truth to define absolute success for limitation to within-experiment comparison winners.

Device Placement
Inconsistent placements across datasets are inconsistent, while our collected data only has fixed positions with no variation.

# Future development:
Activity Expansion:
Exploration on other activities beyond current benchmarks to broaden model utility.

Population Diversity:
Investigation on variables which are more closely, including disabilities, body type, sex, and BMI.

IMU Optimization:
Exploration on additional IMU positions to identify optimal sensor placement.

Evaluation Standardization:
Developement on a unified, quantifiable evaluation framework for assessing pretraining quality and model robustness that generalizes across devices, datasets, and deployment scenarios.




