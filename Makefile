build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp borb --models ihf lr nb mlp irf --start 2000 --end 2050
	jitsdp orb --start 2000 --end 2050

clean:
	rm -rf models/ logs/ tests/logs