# SplitRAG

This repository contains the code, anonymized data, prompts, model outputs, evaluation results, and supplementary diagnostic records used in the paper:

**SplitRAG: Step-Level Evidence Isolation for Resource-Constrained Knowledge Graph Question Answering**

The repository is intended primarily to document the experimental materials reported in the manuscript and supplementary information. The code reflects the research implementation used in the experiments; local model paths and output directories may need to be adjusted before reuse.

## Repository structure

```text
SplitRAG/
├── README.md
├── requirements.txt
│
├── src/
│   ├── pipeline/
│   │   └── prepare_reasoning_inputs.py
│   ├── inference/
│   │   ├── local_inference.py
│   │   └── api_inference.py
│   ├── evaluation/
│   │   └── evaluate_answers.py
│   ├── analysis/
│   │   └── statistical_tests.py
│   └── finetuning/
│       └── train_lora.py
│
├── datasets/
│   ├── internal/
│   │   ├── anonymized_kg.csv
│   │   └── benchmark.zip
│   └── finetuning/
│       └── finetuning_records.zip
│
├── models/
│   └── lora_adapters/
│       ├── Llama-3.2-1B-Instruct/
│       └── Qwen3.5-0.8B/

│
├── results/
│   └── internal/
│       ├── prompts.zip
│       ├── model_outputs.zip
│       └── summary_scores/
│           ├── 1p_scores.xlsx
│           ├── 2p_scores.xlsx
│           ├── 2i_scores.xlsx
│           ├── 2u_scores.xlsx
│           └── 3i_scores.xlsx
│
└── experiments/
    ├── context_expansion/
    ├── dynamic_2p/
    ├── finetuning_overlap/
    ├── identifier_remapping/
    ├── intermediate_evidence/
    ├── structural_sensitivity/
    ├── symbolic_verification/
    ├── cost_analysis/
    ├── fb15k237/
    └── kqapro/
```

Large collections of per-query prompts, outputs, and scores are stored as ZIP archives to avoid placing hundreds of thousands of small files directly in the Git repository. The internal directory structure of each archive is preserved.

## Core implementation

The main shared scripts are located in `src/`.

- `src/pipeline/prepare_reasoning_inputs.py`  
  Constructs reasoning inputs from the internal benchmark, including candidate evidence retrieval, query decomposition, prompt construction, and step-level evidence allocation for SplitRAG.

- `src/inference/local_inference.py`  
  Runs local-model inference for the main experiments.

- `src/inference/api_inference.py`  
  Runs inference with externally hosted API models. API credentials are not included.

- `src/evaluation/evaluate_answers.py`  
  Computes answer-set evaluation scores.

- `src/analysis/statistical_tests.py`  
  Performs the paired statistical analyses reported in the supplementary material.

- `src/finetuning/train_lora.py`  
  Fine-tunes the two small local models with LoRA.

## Internal knowledge graph and benchmark

`datasets/internal/` contains the anonymized internal data used in the controlled experiments.

- `anonymized_kg.csv`: anonymized knowledge-graph triples.
- `benchmark.zip`: structured benchmark instances covering `1p`, `2p`, `2i`, `2u`, and `3i`.

The benchmark records contain the structured queries and gold answers used in the experiments. The numbered instance files are preserved inside the archive so that they remain aligned with the corresponding prompts and model outputs.

## Main experimental records

`results/internal/` contains the records from the main internal-benchmark experiments.

- `prompts.zip`: prompts for NL/RAG/Decomp/SplitRAG/CoT/FS-CoT settings as applicable.
- `model_outputs.zip`: outputs from the local base models, fine-tuned models, and API-model experiments.
- `summary_scores/`: query-type-level score files used for the main result tables.

The archives preserve the original model / strategy / query-type directory hierarchy.

## Fine-tuning

`datasets/finetuning/finetuning_records.zip` contains the 25,000 RAG-formatted fine-tuning examples used for Qwen3.5-0.8B and Llama-3.2-1B-Instruct.

The resulting LoRA adapters are provided in:

```text
models/lora_adapters/
├── Llama-3.2-1B-Instruct/
└── Qwen3.5-0.8B/
```

Each directory contains `adapter_config.json` and `adapter_model.safetensors`.

## Supplementary diagnostic experiments

### Controlled context expansion

`experiments/context_expansion/`

Contains the distractor-generation script and the prompt, output, and score records used to evaluate evidence exposure under increasing context noise.


```text
generate_context_distractors.py
records.zip
```

### Dynamic 2p error propagation

`experiments/dynamic_2p/`

Contains the prediction-conditioned dynamic 2p experiments and the executor diagnostic.

```text
run_dynamic_llm.py
run_dynamic_executor.py
dynamic_llm_records.zip
dynamic_executor_records.zip
```

### Fine-tuning overlap audit

`experiments/finetuning_overlap/`

Contains the script and summary used to check direct overlap between fine-tuning and evaluation data.

```text
audit_overlap.py
summary.json
```

### Identifier-remapping control

`experiments/identifier_remapping/`

Contains the identifier-remapping code, remapping audit, and all remapped prompts, outputs, and scores.

```text
remap_identifiers.py
audit_remapping.py
summary.json
records.zip
```

### Intermediate evidence preservation and recovery

`experiments/intermediate_evidence/`

Contains the per-query Decomp/SplitRAG intermediate predictions, gold intermediate answers, and analysis used for projection F1, recall, branch containment, all-branch containment, and conditional recovery.

```text
analyze_intermediate_evidence.py
sample_level.csv
records.zip
```

### Structural sensitivity

`experiments/structural_sensitivity/`

Contains the relation-corruption and topology-confusion experiments.

```text
generate_perturbations.py
records.zip
```

### Symbolic execution and evidence-sufficiency audit

`experiments/symbolic_verification/`

Contains both the full-KG symbolic execution check and the structured-input evidence-sufficiency audit.

```text
full_kg_executor.py
full_kg_executor_results.csv
input_sufficiency_audit.py
input_sufficiency_summary.csv
input_sufficiency_records.zip
```

### Cost analysis

`experiments/cost_analysis/`

Contains the script and summary used to calculate the token- and call-based inference-cost indicators reported in the manuscript.

```text
analyze_cost.py
summary.json
```

## External validation

### FB15k-237

`experiments/fb15k237/`

Contains the sampled structured benchmark used in the study, entity/relation mappings, generated prompts/evidence, model outputs, scores, and the analysis report.

```text
entity_id_map.pkl
relation_id_map.pkl
analysis_report.md
records.zip
```

### KQA Pro

`experiments/kqapro/`

Contains the retained KQA Pro questions, semantic-parser training/validation records, parser outputs, structured-query conversion, oracle-parse and predicted-parse prompts, model outputs, and evaluation results.

The main scripts are:

```text
filter_supported_queries.py
build_parser_dataset.py
train_parser.py
run_parser_validation.py
convert_parser_output.py
build_kqapro_prompts.py
```

Small JSON files describing the retained evaluation set and parser results are kept uncompressed for inspection. Large prompt / answer / score collections are stored in `inference_records.zip`.

The trained KQA Pro parser adapter is larger than GitHub's normal per-file size limit and should be uploaded as a **GitHub Release asset** (or managed through Git LFS). Its release asset should contain:

```text
adapter_config.json
adapter_model.safetensors
```

## Notes on archived records

The repository contains a large number of per-query experimental files. To keep the Git repository manageable:

1. source code, summary files, CSV/JSON analysis outputs, and small metadata files are kept directly visible;
2. large collections of numbered `.txt` prompt/output/score files are stored in ZIP archives;
3. numbered files inside archives are not renamed because their indices align the benchmark instance, prompt, prediction, and score;
4. trained model weights larger than GitHub's normal file-size limit are stored as GitHub Release assets or with Git LFS.

## API models

API-model scripts require user-provided credentials and endpoint configuration. No API keys are included in this repository.

## Citation

If you use this repository, please cite the associated paper:

> SplitRAG: Step-Level Evidence Isolation for Resource-Constrained Knowledge Graph Question Answering.

