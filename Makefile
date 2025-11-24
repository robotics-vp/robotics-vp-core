# Robotics V-P Economics Model Makefile

PYTHON := python3
SCRIPTS := scripts

.PHONY: all train_stage6 demo video audit clean

all: train_stage6

# Train all Stage 6 components
train_stage6:
	@echo "Starting Stage 6 Training..."
	$(PYTHON) $(SCRIPTS)/train_vision_backbone_real.py --use-mixed-precision --epochs 10
	$(PYTHON) $(SCRIPTS)/train_sima2_segmenter.py --use-mixed-precision --epochs 10
	$(PYTHON) $(SCRIPTS)/train_spatial_rnn.py --use-mixed-precision --epochs 10
	$(PYTHON) $(SCRIPTS)/train_hydra_policy.py --use-mixed-precision --max-steps 100

# Run demo in simulation
demo:
	@echo "Running Demo in Simulation..."
	$(PYTHON) $(SCRIPTS)/run_demo_in_sim.py --task-id drawer_open --use-mixed-precision --num-episodes 5

# Export demo video (requires frames from demo run)
video:
	@echo "Exporting Demo Video..."
	# Assuming demo saves frames to results/demo_runs/frames
	$(PYTHON) $(SCRIPTS)/export_demo_video.py --input-dir results/demo_runs/frames --output-file results/demo.mp4

# Audit codebase
audit:
	@echo "Auditing Codebase..."
	@echo "Checking for untracked files..."
	git status
	@echo "Running smoke tests..."
	$(PYTHON) $(SCRIPTS)/smoke_test_amp_training.py
	$(PYTHON) $(SCRIPTS)/smoke_test_checkpointing.py

# Clean results
clean:
	rm -rf results/training_logs/*
	rm -rf results/checkpoints/*
