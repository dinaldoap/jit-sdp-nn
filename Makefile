build: clean format test run

format:
	autopep8 --in-place --recursive jitsdp setup.py

test: 
	pytest tests

run:
	jitsdp borb --model ihf --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model lr --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model nb --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model mlp --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model irf --start 2000 --end 2050 --experiment-name dev
	jitsdp orb --start 2000 --end 2050 --experiment-name dev
	jitsdp tuning --start 0 --end 1 --cross-project 0 1 --validation-end 5000 1000
	jitsdp testing --start 0 --end 1 --cross-project 0 1 --testing-start 5000 --tuning-experiment-name dev --no-validation
	jitsdp report --start 0 --end 1 --cross-project 0 1 --tuning-experiment-name dev --testing-experiment-name dev --no-validation

clean:
	rm -rf models/ logs/ tests/logs
	rm -rf data/joblib/jitsdp/data/load_runs/67034ca7d7c3020626a5784042924e95/
