# IC-DST: In-Context Learning for Dialogue State Tracking

This is the pytorch implementation of
**In-Context Learning for Dialogue State tracking**. 

[**Yushi Hu**](https://yushi-hu.github.io/), Chia-Hsuan Lee, Tianbao Xie, Tao Yu, Noah A. Smith, and Mari Ostendorf. 
[[PDF]](https://arxiv.org/abs/2203.08568)

Please cite with this bibtex:
<pre>
@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
</pre>

## Environment
Besides PyTorch and Huggingface transformers, install the other requirements by
```console
pip install -r requirements.txt
```
Set up CodeGen by
```console
./install_codegen.sh
```

Put your OpenAI API key in `config.py` to use Codex.
Skip this step if you only want to run on GPT-Neo and CodeGen.

## Data
We follow the pipeline of [MultiWoz 2.4 repo](https://github.com/smartyfh/MultiWOZ2.4/) for data preprocessing.
We modified a bit to unify the ontology between MultiWOZ 2.1 and 2.4
To download and create the dataset
```console
cd data
python create_data.py --main_dir mwz21 --mwz_ver 2.1 --target_path mwz2.1  # for MultiWOZ 2.1
python create_data.py --main_dir mwz24 --mwz_ver 2.4 --target_path mwz2.4  # for MultiWOZ 2.4
```

### preprocess the dataset
Run the following script to sample and preprocess the few-shot and full-shot training sets, dev set and test set. 
For few-shot experiments, the retriever is trained on the selection pool. So we have save the selection pool for each of the experiment.
`data/sample.py` samples and processes the training sets.
All the processed data will be saved in the `data` folder.
```console
./preprocess.sh
```

If you want to sample your own training set, follow the following example:
to sample a 5% training set
```console
python sample.py --input_fn mwz2.1/train_dials.json --target_fn mw21_5p_train_v4.json --ratio 0.05 --seed 0
```


## Retriever
The trained retrievers are saved in `retriever/expts` folder. Each subfolder is a trained retriever.

### retriever quickstart
If you want to skip the retriever finetuning etc. part, just download one of our retriever finetuned on 5% training set and try it!
Download and unzip [https://drive.google.com/file/d/12iOXLyOxvVuepW7h8h7zNj1tdIVNU76F/view?usp=sharing](https://drive.google.com/file/d/12iOXLyOxvVuepW7h8h7zNj1tdIVNU76F/view?usp=sharing), put the folder in `retriever/expts`.

## In-Context Learning Experiments

Given a selection pool and a retriever, to run an in-context learning experiment, for example, using `data/mw21_5p_train_v2.json` as selection pool, use `retriever/expts/mw21_5p_v2` as the retriever (already trained with the selection pool), and saving all the results and running logs to `expts/codex_5p_v2`, and evaluate on the MultiWOZ 2.1 test set:
```console
python run_codex_experiment.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_dir expts/codex_5p_v2  \
      --mwz_ver 2.1
```
Notice that this will generate a json file `expts/codex_5p_v2/running_log.json`. This files contains all the prompts and codex completion, and can be used in further analysis. We have put the sample running log there for analysis.

To run the experiment with GPT-Neo 2.7B, use `run_gpt_neo_experiment.py` rather than `run_codex_experiment.py`. To run with CodeGen-Mono 2.7B, use `run_codegen_experiment.py`. Everything else is the same.

### Analyze using the running log

Notice that the only difference between MultiWOZ 2.1 and 2.4 are the labels of dev and test set. So, there is no need to run the same experiment again for 2.1 and 2.4. Instead, we can get the MultiWOZ 2.4 scores with the running log on MultiWOZ 2.1. Continue with the above example, we can get 2.4 scores by
```console
python evaluate_run_log.py --running_log expts/codex_5p_v2/running_log.json --test_fn data/mw24_100p_test.json --mwz_ver 2.4
```

### Zero-shot experiment
Run the zero-shot experiment on MultiWOZ 2.1 by
```console
python run_zeroshot_codex_experiment.py --output_dir ./expts/zero-shot --mwz_ver 2.1
```

Get the per-domain result on MultiWOZ 2.1 by
```console
python evaluate_run_log_by_domain.py --running_log expts/codex_5p_v2/running_log.json --test_fn data/mw24_100p_test.json --mwz_ver 2.1
```

To get result on MultiWOZ 2.4, change to `--mwz_ver 2.4`.
