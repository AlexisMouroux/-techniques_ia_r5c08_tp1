import glob
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image


def list_files(directory):
    return glob.glob(directory + '/**/*', recursive=True)



# utilisation d'un model pre-entraine pour la classification d'image nom = ResNet101
# test avec une image
# on a besoin a la fois de l'archi du model et des poids

# get pre-trained model weights
weights = ResNet101_Weights.DEFAULT

# le model a des infos dans ses meta data
# categories donne les classes que le model sait classer.
# On a 1000 classes chacune a un id qui donne sa place dans le tensor de sortie
print('nombre de classes que le model sait classifier = {}'.format(len(weights.meta["categories"])))
print('nom des classes d\'id entre 5 et 12 : {}'.format(weights.meta["categories"][5:12]))

# le resnet attend que les images en entree soient des tensor normalised avec comme taille 3 x 234 x 234 => C x H x W
# C = channel : 3 car image couleur RGB
# H = hauteur et W = largeur : 224 x 224
# la class Resnet101 propose directement le pretraitement necessaire pour pouvoir appeler le model
preprocess = weights.transforms()

# instanciation du model avec les poids et passage en model eval => notre but est l'utilisation pas le training pour le moment
mon_model = resnet101(weights=weights)

# le model est au format pytorch et est un graph de calcul
#print('model = {}'.format(mon_model))

# on met le model en mode eval pour eviter que les poids ne soient modifies
mon_model.eval()

# recuperation de l'image de test que l'on veut faire classer par le model
# get all file from a specific repository
directory = './data/test_images/'
files = list_files(directory=directory)
for image in files:
    mon_image = Image.open(image)
    image_name = image.split('/')[-1]

    # application du preprocessing a l'image de test
    image_preprocessed = preprocess(mon_image)
    # check du shape de sortie
    print('shape de l  image apres le preprocess {}'.format({image_preprocessed.shape}))

    # formatage de l'input pour quelle passe bien dans le model
    batch = image_preprocessed.unsqueeze(0)
    print('shape de l input du model {}'.format({batch.shape}))

    # appel du forward du model : resultat = prediction de la classe format = brut
    prediction_non_formatee = mon_model(batch)

    # faire un print du model => regarder comment pytorch représente le model sous forme de graph de calcul

    # analyser le shape de la sortie du model

    # trouver quelle est la prédiction du model => la classe la plus probable

    # trouver le nom de la classe prédite

