# Desafio de Projeto: Treinamento de Rede YOLO com COCO

## 📌 Descrição
Este projeto tem como objetivo criar uma base de dados a partir do dataset COCO, realizar a rotulagem das imagens e treinar a rede YOLO usando Google Colab. Para otimizar o tempo de execução, será utilizada apenas uma parcela do dataset COCO e será aplicado Transfer Learning.

## 🛠️ Tecnologias Utilizadas
- Google Colab
- Python
- Darknet
- YOLO (You Only Look Once)
- Dataset COCO

## 📂 Organização das Pastas
```
/content/darknet/data/coco_subset/
│── images/
│   ├── train/  # Imagens para treino
│   ├── val/    # Imagens para validação
│── labels/
│   ├── train/  # Anotações para treino
│   ├── val/    # Anotações para validação
│── annotations/ # Arquivo JSON de anotações COCO
│── coco.data   # Configuração do dataset para YOLO
│── coco.names  # Lista de classes utilizadas
│── train.txt   # Lista de imagens para treino
│── val.txt     # Lista de imagens para validação
```

## 🚀 Passo a Passo

### 1️⃣ Criar as Pastas e Baixar os Dados do COCO
```python
!mkdir -p /content/darknet/data/coco_subset/images/train
!mkdir -p /content/darknet/data/coco_subset/images/val
!mkdir -p /content/darknet/data/coco_subset/labels/train
!mkdir -p /content/darknet/data/coco_subset/labels/val

!wget http://images.cocodataset.org/zips/train2017.zip -O /content/darknet/data/coco_subset/train2017.zip
!unzip -q /content/darknet/data/coco_subset/train2017.zip -d /content/darknet/data/coco_subset/images/train/

!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /content/darknet/data/coco_subset/annotations.zip
!unzip -q /content/darknet/data/coco_subset/annotations.zip -d /content/darknet/data/coco_subset/
```

### 2️⃣ Criar Arquivo de Configuração `coco.data`
```python
coco_data_content = """
classes= 4
train  = /content/darknet/data/coco_subset/train.txt
valid  = /content/darknet/data/coco_subset/val.txt
names  = /content/darknet/data/coco_subset/coco.names
backup = /content/darknet/backup/
"""
with open("/content/darknet/data/coco_subset/coco.data", "w") as f:
    f.write(coco_data_content)
```

### 3️⃣ Filtrar Imagens do COCO e Criar `train.txt` e `val.txt`
```python
import json
import random
import shutil
import os

annotation_path = "/content/darknet/data/coco_subset/annotations/instances_train2017.json"
selected_classes = ["person", "car", "dog"]

with open(annotation_path, "r") as f:
    coco_data = json.load(f)

category_id_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
filtered_images = set()
for annotation in coco_data["annotations"]:
    if category_id_map.get(annotation["category_id"]) in selected_classes:
        filtered_images.add(annotation["image_id"])

filtered_images = list(filtered_images)
random.shuffle(filtered_images)
filtered_images = filtered_images[:500]

image_source_folder = "/content/darknet/data/coco_subset/images/train2017/"
image_dest_folder = "/content/darknet/data/coco_subset/images/train/"
if not os.path.exists(image_dest_folder):
    os.makedirs(image_dest_folder)

for img_id in filtered_images:
    img_name = f"{str(img_id).zfill(12)}.jpg"
    src = os.path.join(image_source_folder, img_name)
    dst = os.path.join(image_dest_folder, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)

train_txt_path = "/content/darknet/data/coco_subset/train.txt"
val_txt_path = "/content/darknet/data/coco_subset/val.txt"
image_folder = "/content/darknet/data/coco_subset/images/train/"
train_images = filtered_images[:400]
val_images = filtered_images[400:]

with open(train_txt_path, "w") as f:
    f.write("\n".join([image_folder + f"{str(img_id).zfill(12)}.jpg" for img_id in train_images]))

with open(val_txt_path, "w") as f:
    f.write("\n".join([image_folder + f"{str(img_id).zfill(12)}.jpg" for img_id in val_images]))
```

### 4️⃣ Baixar Modelo YOLO Pré-Treinado
```python
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.conv.137 -O /content/darknet/yolov4.conv.137
```

### 5️⃣ Iniciar o Treinamento com Transfer Learning
```python
!./darknet detector train /content/darknet/data/coco_subset/coco.data cfg/yolov4.cfg /content/darknet/yolov4.conv.137 -dont_show -map
```

## 📌 Observações
- **Transfer Learning:** O modelo pré-treinado `yolov4.conv.137` é usado para otimizar o treinamento.
- **Subconjunto do COCO:** Apenas 500 imagens foram usadas para reduzir o tempo de treinamento.
- **Erros Comuns:** Se houver erro no número de classes, verifique `coco.data` e `coco.names`.


