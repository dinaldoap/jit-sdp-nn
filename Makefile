build: clean format test run

format:
	autopep8 --in-place --recursive jitsdp setup.py

test: 
	pytest tests

run:
	jitsdp borb --model ihf --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model ihf --start 2000 --end 2050 --experiment-name dev --rate-driven 1
	jitsdp borb --model lr --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model nb --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model mlp --start 2000 --end 2050 --experiment-name dev
	jitsdp borb --model irf --start 2000 --end 2050 --experiment-name dev
	jitsdp orb --start 2000 --end 2050 --experiment-name dev
	jitsdp orb --start 2000 --end 2050 --experiment-name dev --rate-driven 1
	jitsdp tuning --start 0 --end 1 --cross-project 0
	jitsdp testing --start 0 --end 1 --cross-project 0 --tuning-experiment-name dev --no-validation
	jitsdp report --start 0 --end 1 --cross-project 0 --tuning-experiment-name dev --testing-experiment-name dev --no-validation

clean:
	rm -rf models/ logs/ tests/logs