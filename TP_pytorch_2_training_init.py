import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet101_Weights, resnet101
from tqdm import tqdm

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# utilisation d'un model pre-entraine pour la classification d'image nom = ResNet101
# finetuning pour avoir en sortie une classification binaire : chat / pas chat
# utilisation d'un dataset fait a partir de photos reparties dans des repertoires differents

def check_dir():
    # Print the paths to verify
    print(f"Data directory: {data_dir}")
    print(f"Test data directory: {data_test_dir}")
    print(f"Save model path: {save_model_path}")
    # Check if directories exist
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    print(f"Test data directory exists: {os.path.exists(data_test_dir)}")
    print(f"Save model directory exists: {os.path.exists(os.path.dirname(save_model_path))}")


#------------------------------
# parametres utilisés dans ce fichier
ratio = 0.2 #ratio pour la taille du dataset de validation par rapport a la taille du dataset d'entrainement
batch_size = 4 #taille du batch pour l'entrainement
optimizer_lr = 3e-4 #learning rate de l'optimiseur
optimizer_weight_decay = 0.0001 #weight decay de l'optimiseur
nb_epoch =3 #nombre d'epoch pour l'entrainement
use_saved_model = False

# Construct relative paths
data_dir = os.path.join(base_dir, './images')
data_test_dir = os.path.join(base_dir, './images/cat')
save_model_path = os.path.join(base_dir, './models/cat_model_freeze.pth')
save_metrics_path = os.path.join(base_dir, './metrics')

check_dir()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_tensor_as_image(tensor, filename):
    # methode pour prendre un tensor qui represente une image et la sauvegarder en tant qu'image
    # Convert the tensor to a NumPy array
    array = tensor.permute(1, 2, 0).numpy()

    # Normalize the array to the range [0, 255]
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(array)

    # Save the image as a PNG file
    image.save(filename)


def instanciate_model():
    # 1 - instanciation du modele : resnet101 => preparation pour le transfert learning
    weights = ResNet101_Weights.DEFAULT
    print('nombre de classes que le model sait classifier = {}'.format(len(weights.meta["categories"])))
    preprocess = weights.transforms()
    mon_model = resnet101(weights=weights)
    # freeze layers : car on veut faire du transfert learnin : pas de changement pour les layer d extraction de feature
    for param in mon_model.parameters():
        param.requires_grad = False
    # ajouter un layer fully connected a la fin du resnet pour ne pas avoir 1000 classes en sortie mais 2 : chat /pas chat
    in_features = mon_model.fc.in_features
    mon_model.fc = torch.nn.Linear(in_features=in_features, out_features=2)
    # init des poids et biais du layer fully connected
    mon_model.fc.weight = nn.init.normal_(mon_model.fc.weight, mean=0.0, std=0.01)
    mon_model.fc.bias = nn.init.zeros_(mon_model.fc.bias)
    mon_model = mon_model.to(device)
    return mon_model, preprocess


def get_dataset():
    # transforme les deux repos avec les images en dataset labelisé
    # 2 - creation d'un dataset chat / pas chat a partir du repertoire images
    dataset = ImageFolder(root=data_dir, transform=preprocess)
    valid_size = int(len(dataset) * ratio)
    train_size = int(len(dataset) - valid_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_data_loader, validate_data_loader


def get_optimizer_loss_function():
    # l'optimizer est un algo qui permet de mettre a jour les poids du model de maniere a minimiser la loss function
    # il utilise la backpropagation pour calculer les gradients et le learning rate pour mettre a jour les poids
    # il existe plusieurs optimiseur : Adam, SGD, RMSprop, ...le plus classique est Adam
    # isntanciation de l'Optimiser :
    optimiser = Adam(mon_model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
    # Linstanciation de la fonction de loss
    # c'est le calcul de la difference entre la prediction et la classe reelle, il existe plusieurs loss function
    # ici on utilise CrossEntropyLoss qui est la plus classique pour la classification
    loss_fn = CrossEntropyLoss()
    return optimiser, loss_fn


def check_result_image_loader(train_data_loader):
    # pour verifier ce que nous fait le ImageLoader : on sauvegarde les 4 premieres images du dataset
    # pour les label on a un int si cat alors valeur = 0 sinon = 1 (si on avait une troisieme classe on aurait 2)
    for images, labels in train_data_loader:
        # shape = 4 ,3 , 224, 224 = batch de 4 images RGB de taille 224 x 224
        for i in range(4):
            save_tensor_as_image(images[i], f'label_{labels[i]}_{i}.png')
        break


# main du TP
if __name__ == '__main__':

    # on a besoin d'un model
    mon_model, preprocess = instanciate_model()

    # on a besoin d'un optimiseur et d'une fonction de cout
    optimiser, loss_fn = get_optimizer_loss_function()

    # on a besoin d'un dataset et qu'il soit divisé en 2 : training et validation
    train_data_loader, validate_data_loader = get_dataset()

    # on a besoin de la taille du dataset pour calculer les metrics
    len_train = len(train_data_loader)
    training_accuracy_tab = []
    training_loss_tab = []

    len_val = len(validate_data_loader)
    validation_accuracy = []
    validation_loss_tab = []

    # on a besoin de verifier que le ImageLoader fonctionne bien
    check_result_image_loader(train_data_loader=train_data_loader)

    # ------------------------------
    # 4 - entrainement du model : boucle sur un nombre d'epoch
    # pour chaque epoch il y a une partie train et une partie validation
    for epoch in range(nb_epoch):
        start = time()

        # initialisation des variables pour les metriques accuracy du training et du test
        nb_prediction_ok_current_epoch_training = 0
        nb_prediction_ko_current_epoch_validation = 0
        total_training_loss = 0
        total_validation_loss = 0

        # mise en mode training du model
        mon_model.train()
        # boucle sur les batchs du dataset on utilise tqdm pour avoir une barre de progression
        with tqdm(train_data_loader, unit="batch") as tepoch:
            for xtrain, ytrain in tepoch:
                # shape = 4 ,3 , 224, 224 = batch de 4 images RGB de taille 224 x 224
                #print(xtrain.shape)
                #print(ytrain)

                # initialisation des gradients a zero pour ne pas les accumuler
                optimiser.zero_grad()

                # passage des donnees sur le device CPU ou CUDA si on a un GPU
                xtrain = xtrain.to(device)
                # appel du forward du model => calcul de la prediction
                train_prob = mon_model(xtrain)

                # calcul de la loss fonction entre le resultat de la prediction et la classe reelle
                loss = loss_fn(train_prob, ytrain)
                # realise la backprogagation pour calculer les gradients
                loss.backward()
                # realise l'optimisation des poids du model
                optimiser.step()

                # Accumulate the loss
                total_training_loss += loss.item()

                # calcul de la metrique accuracy pour le training
                # sortie de la prediction = tensor de taille 4 x 2 : 4 images par batch et 2 classes
                # on prend la classe avec la plus grande proba
                # l indice de la classe avec la plus grande proba est la prediction

                # tensor([[ 0.0613, -0.1003],
                #         [ 0.1708,  0.1259],
                #         [ 0.0537, -0.1559],
                #         [ 0.2059, -0.0064]], grad_fn=<AddmmBackward0>)
                # pour l'image 1 du batch la classe la plus probable est la 0 (chat) avec une proba de 0.0613
                train_pred = torch.max(train_prob, 1).indices
                # train_pred = tensor([1, 0, 1, 1])
                # ytrain = tensor([1, 0, 0, 0])
                # on compare la prediction avec la classe reelle
                # on a 2 cas ou la prediction est correcte => on ajoute 2 au nombre de prediction correcte
                nb_prediction_ok_current_epoch_training += int(torch.sum(train_pred == ytrain))

            #accuracy pour l'epoque en cours = nombre de prediction correcte / nombre total de prediction = taille du dataset de training
            nombre_total_image_training_set = len_train*batch_size
            training_accuracy_current_epoch = nb_prediction_ok_current_epoch_training / nombre_total_image_training_set
            training_accuracy_tab.append(training_accuracy_current_epoch)
            #To calculate the training loss for an epoch, you need to accumulate the loss for each batch and then average it over the number of batches
            # loss est un tensor => on le transforme en float pour l'ajouter a la liste
            # Calculate the average loss for the epoch
            average_training_loss = total_training_loss / nombre_total_image_training_set
            training_loss_tab.append(average_training_loss)

        # Evaluate
        # mise en mode evaluation du model
        mon_model.eval()
        # ecrire la partie evaluation du model
        # utilisation du dataset de validation : autre morceau du dataset

        # accuracy pour l'epoque en cours = nombre de prediction correcte / nombre total de prediction = taille du dataset de validation
        nombre_total_image_validation_set = len_val*batch_size
        # calculer l'accuarcy pour le dataset de validation pour l'epoque en cours
        average_validation_loss = total_validation_loss / nombre_total_image_validation_set
        validation_loss_tab.append(average_validation_loss)

        # metrics de fin d'epoque
        end = time()
        duration = (end - start) / 60
        print(f"Epoch: {epoch}, Time: {duration}, Loss training : {average_training_loss}, Loss validation: {average_validation_loss}\n,  accuracy training : {training_accuracy_current_epoch}, accuracy validation : {ep_eval_acc}\n")


    # plot accuracy per epoch et loss per epoch


    # fin du training on sauvegarde le model en local
    torch.save(mon_model, save_model_path)
    print(f"Model saved to {save_model_path}")




