build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp run --models lr nb mlp rf --orb 1 --f_folds .1

clean:
	rm -rf models/ logs/ tests/logs