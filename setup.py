from setuptools import setup, Extension

setup(
    name="linear_tokenizer",
    ext_modules=[
        Extension("_features", sources=["_features.c"]),
    ],
)
