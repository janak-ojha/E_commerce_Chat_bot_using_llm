from setuptools import find_packages, setup

setup(
    name="Ecommercebot",
    version="0.0.1",
    author="janak",
    author_email="janakrajojha230@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain ','datasets','pypdf','python-dotenv','flask','transformers','huggingface_hub']
)