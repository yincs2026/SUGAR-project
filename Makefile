.PHONY: install data offline train baselines ablation e3 e4 e5 e6 e7 e8 e9 test report all
DATASET ?= beauty

install:
	pip install -e .

data:
	python scripts/prepare_data.py --config configs/datasets/beauty.yaml
	python scripts/prepare_data.py --config configs/datasets/yelp_filtered.yaml

offline:
	python scripts/run_offline.py --phase all --dataset $(DATASET)

train:
	python scripts/train.py --model sugar --dataset beauty --tag sugar_beauty

baselines:
	for M in gru4rec sasrec bert4rec unisrec; do \
	  python scripts/train_baseline.py --model $$M --dataset beauty --tag $${M}_beauty; \
	done

ablation:
	python scripts/run_ablation.py --dataset beauty --variants a1 a2 a3 a4

e3:
	python scripts/inject_drift.py --dataset beauty
	python scripts/run_offline.py --phase m2m3 --dataset beauty_drift

e4:
	for P in 0.2 0.4 0.6 0.8; do \
	  python scripts/corrupt_profiles.py --dataset beauty --p $$P --eval_model sugar --ckpt_tag sugar_beauty; \
	  python scripts/corrupt_profiles.py --dataset beauty --p $$P --eval_model a3_fixed_gate --ckpt_tag a3_a3_fixed_gate_beauty; \
	done
	python scripts/vary_lag.py --dataset beauty --ckpt_tag sugar_beauty

e5:
	python scripts/swap_graph.py --dataset yelp_filtered --ckpt_tag sugar_yelp_filtered

e6:
	python scripts/schema_stats.py --dataset beauty --judge 200

e7:
	python scripts/sensitivity.py --dataset beauty --param offline.window_size_k --values 5 10 20
	python scripts/sensitivity.py --dataset beauty --param model.lambda_align --values 0 0.01 0.1

e8:
	python scripts/cost_profile.py --dataset beauty

e9:
	python scripts/case_study.py --dataset beauty --ckpt_tag sugar_beauty

test:
	pytest tests/ -v

report:
	python analysis/make_tables.py
	python analysis/make_plots.py

all:
	bash run_all.sh all