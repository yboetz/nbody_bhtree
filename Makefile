clean:
	rm -f -r build/
	rm -f -r src/__pycache__/
	rm -f lib/octree.cp*

clean-build:
	rm -f -r build/

.PHONY: build
build: clean
	python setup.py build_ext --inplace
