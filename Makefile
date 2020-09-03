build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp --models bht lr nb mlp rf --start 2000 --end 2050
	baseline --start 2000 --end 2050

clean:
	rm -rf models/ logs/ tests/logs