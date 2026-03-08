# Detecting Manipulative Narratives in Social Media

This repository provides the code for the Bachelor thesis.

## Overview
During the 2022 Russian invasion of Ukraine, Telegram became a crucial plat- form for both information sharing and the spread of propaganda. Its speed, reach, and minimal moderation turned it into a powerful tool not only for communication but also for influence operations targeting civilians. This situation underscored the need for effective methods to detect manipulative narratives in multilingual online environments.This thesis presents one of the top-performing solutions to the UNLP 2025 Shared Task on Detecting Manipulation in Social Media. The task focuses on detecting and classifying rhetorical and stylistic manipulation techniques used to influence Ukrainian Telegram users. For the classification subtask, we fine-tuned the Gemma 2 lan- guage model with LoRA adapters and applied a second-level classifier leveraging meta-features and threshold optimization. For span detection, we employed an XLM- RoBERTa model trained for multi-target, including token binary classification. Our approach achieved 2nd place in classification and 3rd place in span detection.

## Prerequisities
All the required libraries are listed in `requirements.txt`.

## Data
The input data, i.e., training and testing sets, are provided by [UNLP Shared Task](https://github.com/unlp-workshop/unlp-2025-shared-task/tree/main/data). Specifically, they provide [`train.parquet`](https://raw.githubusercontent.com/unlp-workshop/unlp-2025-shared-task/main/data/techniques_classification/train.parquet) with all required columns, indicating manipulative techniques and spans in texts, and [`test.csv`](https://raw.githubusercontent.com/unlp-workshop/unlp-2025-shared-task/main/data/techniques_classification/test.csv) with only texts. The solution for `test.csv` is stored in two separate files: [solution for techniques classification](https://raw.githubusercontent.com/unlp-workshop/unlp-2025-shared-task/main/data/techniques_classification/solution.csv) and [solution for span detection](https://raw.githubusercontent.com/unlp-workshop/unlp-2025-shared-task/refs/heads/main/data/span_detection/solution.csv). 

The code to dowload this data and construct a single testing file is stored in `eda/00-download_data.ipynb` Jupiter notebook.

`eda/` folder also contains three Jupiter notebooks with EDA for training and testing data seperately and when combined into one dataset. Since the distribution of data across both sets was similar, it was more convinient to look at the data as whole. These notebooks present standard analysis like distribution of techniques and languages, as well as named entity recognition (NER) in texts and sentiment analysis. Since the testing data didn't contain information on language of texts, we performed language detection using `langdetect` library; however, due to the instability of recognition results (on average 3-4 samples are assigned different language), we provide `data/test_lang.csv` file, which we used for language senitive analysis, for the accurate reproduction of results.

`data/` folder is the path where the data will be downloaded, also it already contains results of experiments and `test_lang.csv`.

## Experiments
`experiments/` folder contains code to solve two subtasks: `technique_classification/` and `span_detection/`. 

### Technique classification
For technique classification task, we present two baselines: classical and with meta-features.

`02-baseline-classical.ipynb` presents classical baseline, which consists of experiments for combination of two text representation methods (TF-IDF and XLM-RoBERTa-Large embeddings) and two classifiers (CatBoost and LightGBM).

`02-baseline-metafeatures.ipynb` presents Catboost model trained on metafeatures like distance to clusters of trigger words, i.e., words in manipulative spans, and information about similar texts.

Our solution is presented in the following Jupiter notebooks:

- `02_1-prompts_preparation.ipynb` prepares few-shot prompts for causal LM with instruction to identify manipulative techniques in a text and description of manipulation techniques;
- `02_2-features_preparation.ipynb` prepares meta-features also used for the second baseline;
- `02_3-сasual_lm_tunning.ipynb` fine-tunes Gemma 2 with causal LM using LoRA adapter;
- `02_4-sequence_classification_tunning.ipynb` fine-tunes Gemma 2 with sequence classification setup using LoRA adapter, then modifies the results using CatBoost and metafeatures.

The result after two-stage Gemma 2 fine-tuning is stored in `pred_classification_gemma.csv`. The result after further postprocessing with CatBoost can be found in `pred_classification_gemma_post_process.csv`.

### Span detection
`03-baseline-xlm_roberta_large.ipynb` contains baseline for span detection task. The baseline model consists of the XLM-RoBERTa-Large transformer followed by a linear classification head, predicting a binary label (span or non-span) for each token. The result is stored in `data/pred_span_xlm_roberta_large.csv`.

`03-xlm_roberta_large-multitask.ipynb` contains our solution for the task. Here we employ a multi-headed architecture based on the XLM-RoBERTa-Large with two custom classification heads on top: one dedicated to the classification of manipulative techniques (multi-label classification) and the other to span identification. Both heads share a common encoder, allowing the model to benefit from shared representations across tasks. The result is stored in `data/pred_span_xlm_roberta_large_multitask.csv`.

## Results
The visualizations used in the thesis are stored in `reports/figures/`.

### Technique classification

<table border="1" cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th>Model</th>
      <th>Data vectorization</th>
      <th>F1 macro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LightGBM</td>
      <td>TF-IDF</td>
      <td>0.23</td>
    </tr>
    <tr>
      <td>CatBoost</td>
      <td>TF-IDF</td>
      <td>0.18</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>XLM-RoBERTa</td>
      <td>0.15</td>
    </tr>
    <tr>
      <td>CatBoost</td>
      <td>XLM-RoBERTa</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>CatBoost+metafeatures</td>
      <td></td>
      <td>0.43</td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Gemma 2</td>
      <td></td>
      <td>0.45007</td>
    </tr>
    <tr>
      <td>Gemma 2 with post-processing</td>
      <td></td>
      <td>0.45447</td>
    </tr>
  </tbody>
</table>

The results show that Gemma 2-based solutions outperformed the baseline and led to an improved F1 score. While the post-processing step contributed only a small gain, it was crucial for securing a competitive edge in the UNLP Shared Task. As a result, this approach achieved second place in the competition.


### Span detection

<table border="1" cellspacing="0" cellpadding="4">
  <thead>
    <tr>
      <th>Model</th>
      <th>Span-level F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>0.58588</td>
    </tr>
    <tr>
      <td>Two-head transformer</td>
      <td>0.59888</td>
    </tr>
  </tbody>
</table>

We can see that the performance improvement was relatively modest.
Nevertheless, this approach ultimately secured us third place in the competition. These findings suggest that, for practical applications, a simpler baseline approach may be more robust and justified.