# EcoSort Edge-AI

**Vikram Rathod**  
1st-year B.Tech CSE student, NITK

Hi! This is my EcoSort Edge-AI project. It classifies waste images into six categories: **cardboard, glass, metal, paper, plastic, and trash**. I trained the model, exported it to ONNX format, and checked it against the project requirements.

## Project files

    EcoSort-EdgeAI/
    ├── README.md
    ├── requirements.txt
    ├── download_data.py
    ├── train_template.py
    ├── verify_submission.py
    └── model.onnx

## Requirements

- Model size: **1.50 MB or less**
- Parameters: **350,000 or fewer**
- Input shape: **(Batch, 3, 96, 96)**, float32
- Output shape: **(Batch, 6)**, float32
- Target accuracy: **at least 75%**
- CPU latency target: **35 ms or less per image**

## How to run

Run these commands from the `EcoSort-EdgeAI` project folder.

### 1. Install dependencies

    python -m pip install -r requirements.txt

### 2. Prepare the dataset

The data script does not download the dataset. Get the archive from the challenge source and place it here:

    ../data/ecosort_dataset_public.zip

The starter README shared this [dataset link](https://drive.google.com/file/d/12lDsBBYZHPM_a8vV2D4CRgxgPH0VI-ZZ/view?usp=sharing).

Then run:

    python download_data.py

The scripts expect the extracted training and validation images under `../data/public/`, organized into folders by class.

### 3. Train and export the model

    python train_template.py --data_dir ../data/public --epochs 10 --batch_size 32 --lr 0.001 --output model.onnx

This trains the classifier, reports training and validation accuracy, and exports the model as `model.onnx`.

### 4. Verify the model

    python verify_submission.py --model model.onnx --val_dir ../data/public/val

The verifier checks the model size, parameter count, input and output shapes, CPU latency, and local validation accuracy.

## Results

- Validation accuracy: **[enter measured accuracy]%**
- Model size: **[enter measured size] MB**
- Parameter count: **[enter measured count]**
- Average CPU latency: **[enter measured latency] ms/image**
- Verification: **[enter result after running the verifier]**

## What I learned

This project helped me practice preparing image data, training an image classifier, exporting a model to ONNX, and checking its size and inference performance.

## Submission

I submitted the ONNX model and training code according to the instructions provided by the challenge organizers.