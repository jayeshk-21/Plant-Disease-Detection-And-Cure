from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import PDDForm
from .models import PddImages
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from tensorflow.keras import optimizers
import cv2
from . import plant_infection
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0

#D:/Django-Learning/PDD/Plant_disease_detection/

# directory_root = '/content/plantvillage'
width = 256
height = 256
depth = 3


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def detection(image_model):
    new_model = load_model('./PDD.h5')
    opt = optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # distribution
    new_model.compile(loss="binary_crossentropy",
                      optimizer=opt, metrics=["accuracy"])

    filename = 'media/' + str(image_model.picture)

    # image_dir="../input/plantdisease/PlantVillage/Tomato_Spider_mites_Two_spotted_spider_mite/002835d1-c18e-4471-aa6e-8d8c29585e9b___Com.G_SpM_FL 8584.JPG"

    im = convert_image_to_array(filename)
    np_image_li = np.array(im, dtype=np.float16) / 225.0
    npp_image = np.expand_dims(np_image_li, axis=0)

    # npp_image = decode_jpeg(npp_image)
    result = new_model.predict(npp_image)
    print(result)

    itemindex = np.where(result == np.max(result))
    # print("Probability:"+str(np.max(result))+"\n"+label_binarizer.classes_[itemindex[1][0]])

    disease_classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                       'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                       'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                       'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                       'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                       'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                       'Tomato_healthy']

    print(disease_classes[itemindex[1][0]])
    dis_type = disease_classes[itemindex[1][0]]
    value = np.max(result)
    return (value, dis_type)


# Create your views here.

# def home(request):
#    return render(request, "pdd/index.html")

def index(request):
    if request.method == 'POST':
        form = PDDForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = PDDForm()
        # return render(request, 'hotel_image_form.html', {'form' : form})
    return render(request, "pdd\index.html", {'form': form})


def success(request):
    pesticides_PEB = {10:"Avoid irrigation in cool cloudy weather and timely irrigation to allow plants to dry before nightfall. Trichoderma harzianum, T.viride, and Pseudomonas fluorescens are effective against the disease. Protectant fungicides (e.g. maneb, mancozeb, chlorothalonil, and triphenyl tin hydroxide) are also effective."}
    pesticides_PLB = {10:"The severe late blight can be effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days after application of systemic fungicides."}
    pesticides_PBBS = {10:"Actigard (Syngenta Crop Protection) and Quintec (Dow AgroSciences) are also good preventives. A three to five day schedule is adequate for fungicide treatments in most cases."}
    pesticides_TTS = {10:"The strobilurin fungicide azoxystrobin and a combination product of mancozeb and fumoxate provided excellent control of target spot."}
    pesticides_TMV = {10:"Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. ... Monitor the rest of your plants closely, especially those that were located near infected plants. Disinfect gardening tools after every use."}
    pesticides_TYLCV = {10:"Imidacloprid should be sprayed on the entire plant and below the leaves; eggs and flies are often found below the leaves. Spray every 14-21 days and rotate on a monthly basis with Abamectin so that the whiteflies do not build-up resistance to chemicals.Chemicals are most effective when used early in the morning and late in the evening. This is because insects are most active at these times of the day."}
    pesticides_TBS = {10:"kal aana", 20:"Once leaf spot has infected your plants, spray with copper fungicide for 7 to 10 days. After that, continue to treat every 10 days when weather is dry or every five to seven days when weather is wet. Copper fungicide can also be used preventively after sowing seeds but before moving plants into the garden."}
    pesticides_TEB = {10:"Use Bacillus subtilis + Pseudomonas florescence(Multiplex Biojodi), Pseudomonas florescence(ECOMONAS OR BACTIVIPE OR SPOT), Trichoderma viride(SANJEEVINI OR NISARGA OR ECODERMA) Quantity: 20 - 25 gm/L of water *, Colloidal silver and colloidal copper(VIDI Green path) 2 mL/L."}
    pesticides_TLB = {10:"Use fungicide sprays based on mandipropamid, chlorothalonil, fluazinam, mancozeb to combat late blight. Fungicides are generally needed only if the disease appears during a time of year when rain is likely or overhead irrigation is practiced."}
    pesticides_TLM = {10:"Protectant fungicides such as chlorothalonil used in weekly protectant spray programs will help control leaf mold. For organic growers, regular copper applications may help to suppress the disease. Cultivars with resistance to leaf mold have been developed."}
    pesticides_TSLS = {10:"Fungicides containing maneb, mancozeb, chlorothalonil effectively control Septoria leaf spot. Apply at 7 to 10 day intervals throughout the season, mainly during flowering and fruit setting. Follow harvest restrictions listed on the pesticide label."}
    pesticides_TSMTSSM = {10:"The best way to begin treating for two-spotted mites is to apply a pesticide specific to mites called a miticide. Ideally, you should start treating for two-spotted mites before your plants are seriously damaged. Apply the miticide for control of two-spotted mites every 7 days or so."}

    image_model = PddImages.objects.latest('id')
    print("----------", image_model)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", image_model.picture.url)
    value, dis_type = detection(image_model)

    image_model.disease_type = dis_type
    image_model.save()
    per = plant_infection.infec_region(
        'D:/Django-Learning/PDD/Plant_disease_detection/'+image_model.picture.url) #change first hardcoded string acc. to your path

    user_image = image_model.picture
    print(user_image)

    if(dis_type == "Tomato_healthy" or dis_type == "Potato___healthy" or dis_type == "Pepper__bell___healthy"):
        return render(request, "pdd\healthy_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type})
    elif(dis_type == "Potato___Early_blight"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_PEB, 'per': per})
    elif(dis_type == "Potato___Late_blight"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_PLB, 'per': per})
    elif(dis_type == "Pepper__bell___Bacterial_spot"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_PBBS, 'per': per})
    elif(dis_type == "Tomato__Target_Spot"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TTS, 'per': per})
    elif(dis_type == "Tomato__Tomato_mosaic_virus"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TMV, 'per': per})
    elif(dis_type == "Tomato__Tomato_YellowLeaf__Curl_Virus"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TYLCV, 'per': per})
    elif(dis_type == "Tomato_Bacterial_spot"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TBS, 'per': per})
    elif(dis_type == "Tomato_Early_blight"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TEB, 'per': per})
    elif(dis_type == "Tomato_Late_blight"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TLB, 'per': per})
    elif(dis_type == "Tomato_Leaf_Mold"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TLM, 'per': per})
    elif(dis_type == "Tomato_Septoria_leaf_spot"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TSLS, 'per': per})
    elif(dis_type == "Tomato_Spider_mites_Two_spotted_spider_mite"):
        return render(request, "pdd\disease_plant.html", {'image_model': user_image, 'value': value, 'dis_type': dis_type, 'pesticides': pesticides_TSMTSSM, 'per': per})
