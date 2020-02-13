build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp run

clean:
	rm -rf models/ logs/ tests/logs