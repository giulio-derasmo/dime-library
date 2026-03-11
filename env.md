conda create -n dime python=3.10 -y
conda activate dime
conda install conda-forge::faiss
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --upgrade ir_datasets
pip install pandas
pip install -U sentence-transformers
pip install nvitop
pip install safetensors
pip install python-dotenv
pip install ir_measures

conda activate dime
mkdir -p $CONDA_PREFIX/etc/conda/activate.d;
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


## pipelines/... commands
for model in ance contriever tasb; do python pipelines/evaluate.py -c dlhard -m "$model" --save; done

for collection in dl19 dl20 dlhard; do for model in ance contriever tasb; do python pipelines/dime.py -f prf --config configs/prf_k2.yaml --save --overwrite --n_jobs -1 --selector top-alpha -c "$collection" -m "$model" --save; done; done

for collection in dl19 dl20 dlhard; do for model in ance contriever tasb; do python pipelines/dime.py -f gpt --config configs/gpt_chatgpt.yaml --save --overwrite --n_jobs -1 --selector top-alpha -c "$collection" -m "$model" --save; done; done

for collection in dl19 dl20 dlhard; do for model in ance contriever tasb; do python pipelines/dime.py -f prf --config configs/prf_k2.yaml --save --overwrite --n_jobs -1 --selector rdime -c "$collection" -m "$model" --save; done; done

for collection in dl19 dl20 dlhard; do for model in ance contriever tasb; do python pipelines/dime.py -f gpt --config configs/gpt_chatgpt.yaml --save --overwrite --n_jobs -1 --selector rdime -c "$collection" -m "$model" --save; done; done


## report.py commands

> tables/table1.tex
── BASELINE TABLES ────────────────────────────────────────────────────────────

python src/report.py --table performance \
    --models ance contriever tasb \
    --collections dl19 dl20 dlhard \
    --selector top-alpha --alpha 0.8 \
    --filters prf-k2 oracular \
    --measures nDCG@10 AP

→ "This table shows retrieval effectiveness at a fixed alpha=0.8 across
   all models and collections. Useful as a standard performance table
   comparing filters head-to-head with multiple measures side by side."

python src/report.py --table performance \
    --models ance contriever tasb \
    --collections dl19 dl20 dlhard \
    --selector rdime \
    --filters prf-k2 oracular \
    --measures nDCG@10 AP
→ "Same structure but for RDIME results. Drop this next to the alpha=0.8
   table to show RDIME matches fixed-alpha performance."

── RDIME VS TOP-K COMPARISON ──────────────────────────────────────────────────

python src/report.py --table comparison \
    --models ance contriever tasb \
    --collections dl19 dl20 dlhard \
    --filters prf-k2 oracular \
    --topk-alphas 0.4 0.6 0.8 \
    --measure nDCG@10
→ "This is the paper's Table 1. Shows RDIME vs a grid of fixed alphas,
   with delta(%) and retained fraction. One measure, many configs.
   Explicitly named 'comparison' so its purpose is clear."

python src/report.py --table comparison \
    --models ance contriever tasb \
    --collections dl19 dl20 dlhard \
    --filters prf-k2 oracular \
    --topk-alphas 0.4 0.6 0.8 \
    --measure AP
→ "Same comparison table but for AP instead of nDCG@10.
   Goes in the appendix alongside the nDCG@10 version."

 ── ALPHA SWEEP ────────────────────────────────────────────────────────────────

python src/report.py --table sweep \
    --models contriever \
    --collections dl19 dl20 dlhard \
    --filter prf-k2 \
    --measures nDCG@10 AP \
    --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \

→ "This table shows how performance evolves as more dimensions are added
   for Contriever with PRF. Useful for the analysis section — you can
   see where the curve saturates. Multiple measures shown together."

python src/report.py --table sweep \
    --models ance contriever tasb \
    --collections dl19 \
    --filter prf-k2 \
    --measures nDCG@10 \
    --alphas 0.1 0.3 0.5 0.7 0.9 1.0
→ "Coarser alpha grid across all models on one collection.
   Good for a compact appendix table."

── RETAINED DIMENSIONS ────────────────────────────────────────────────────────

python src/report.py --table retained \
    --models ance contriever tasb \
    --collections dl19 dl20 dlhard \
    --filters prf-k2 oracular
→ "This table shows the mean fraction of dimensions retained by RDIME
   per (model, filter, collection). Complements the boxplot figure.
   Reveals which models concentrate signal in fewer dimensions."

── FULL APPENDIX TABLE ────────────────────────────────────────────────────────

python src/report.py --table performance \
    --models ance contriever tasb cocondenser \
    --collections dl19 dl20 dlhard \
    --selector rdime \
    --filters prf-k2 prf-k10 oracular \
    --measures nDCG@10 AP R@1000 RR@10 \
    > tables/performance.tex
→ "Full appendix table with all models, all filters, all four measures
   under RDIME. Wide table — use table* and \small or \footnotesize."