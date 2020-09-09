build: clean format test run

format:
	autopep8 --in-place --recursive .

test: 
	pytest tests

run:
	jitsdp borb --model ihf --start 2000 --end 2050
	jitsdp borb --model lr --start 2000 --end 2050
	jitsdp borb --model nb --start 2000 --end 2050
	jitsdp borb --model mlp --start 2000 --end 2050
	jitsdp borb --model irf --start 2000 --end 2050
	jitsdp orb --start 2000 --end 2050
	jitsdp tuning --start 0 --end 1

clean:
	rm -rf models/ logs/ tests/logs