build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp run --models lr nb mlp rf svm --orb 1 --start 2000 --end 2050

clean:
	rm -rf models/ logs/ tests/logs