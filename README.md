follow these instructions to set up and run the app locally

## create a virtual environment
```bash
python -m venv .venv
```
## activate the virtual environment
```bash
.venv\Scripts\activate.bat
```
## install dependencies
```bash
pip install -r requirements.txt
```

## setup tabpfn token
tabpfn requires a token, said token can be acquired from https://docs.priorlabs.ai/quickstart after getting the token make a directory named .streamlit and make a file named secrets.toml with the following line:
```bash
TABPFN_TOKEN = "INSERT YOUR TOKEN HERE"
```

## models and other files
pre-trained models, label encoder, etc can be accessed in https://drive.google.com/drive/u/0/folders/1J8p8RTHsrZaQXi_frxUd88vRDEDTvRYU 

make sure to put them in the same directory as app.py

or you can train your own model and then export it using https://colab.research.google.com/drive/1bDKbp3I1RptT6iIn4u3a7wvDbZOjhuWR?usp=sharing

if you dont have your own dataset, the dataset used to train the models can be accessed in https://drive.google.com/drive/u/0/folders/1J8p8RTHsrZaQXi_frxUd88vRDEDTvRYU

## gpu acceleration
make sure gpu acceleration is enabled on whatever ide is being used, the app has not been tested without gpu acceleration so errors might occur if gpu acceleration is off

## the app

```bash
streamlit run app.py
```
