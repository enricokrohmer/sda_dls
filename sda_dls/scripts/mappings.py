# List of models from ssb that are summarised as one
# single class. Name on the left is the class name which
# is used in the dataset
SIMILAR_MODELS_MODEL = [
    ["Peugeot_508", "Peugeot_508SW"],
    ["Porsche_Panamera_2010", "Porsche_Panamera_2016"],
    ["Renault_Megane", "Renault_MeganeCabrio"],
    ["Skoda_Rapid", "Skoda_RapidSpaceback"],
    ["Toyota_Prius", "Toyota_PriusPlug-In"],
    ["Toyota_Yaris_2009", "Toyota_Yaris_2011"],
]

# Vehicle Models in both datasets which have
# the same model generation and facelift.
# Keys denote SSB classes and right side ccsv classes
MAPPING_MODEL_YEAR_FACELIFT = MAPPING_MODEL_YEAR_FACELIFT_REVERSED = {
    "Audi_A1": "Audi,Audi A1",
    "Audi_S7": "Audi,Audi A7",
    "Audi_Q3": "Audi,Audi Q3",
    "Dodge_Caliber": "Dodge,Caliber",
    "Jeep_Cherokee": "Jeep,Grand Cherokee",
    "LandRover_RangeRoverSport": "LAND-ROVER,Range Rover",
    "Mazda_CX-5": "MAZDA,Mazda CX-5",
    "Mercedes-Benz_CLS-class": "Benz,Benz CLS Class",
    "Peugeot_508": "Peugeot,Peugeot 508",
    "Porsche_Cayenne": "Porsche,Canyenne",
    "Porsche_Panamera_2010": "Porsche,Panamera",
    "Skoda_Rapid": "Skoda,Rapid",
}

# Vehicle Models in both datasets which have
# the same model generation and same generation
# Used in the paper.
MAPPING_MODEL_YEAR = {
    "Audi_A1": "Audi,Audi A1",
    "Audi_S7": "Audi,Audi A7",
    "Audi_Q3": "Audi,Audi Q3",
    "Dodge_Caliber": "Dodge,Caliber",
    "Hyundai_SantaFe": "Hyundai ,Santafe",
    "Jaguar_XF": "Jaguar,Gaguar XF",
    "Jeep_Cherokee": "Jeep,Grand Cherokee",
    "LandRover_RangeRoverSport": "LAND-ROVER,Range Rover",
    "Lexus_CT": "Lexus,Lexus CT",
    "Mazda_CX-5": "MAZDA,Mazda CX-5",
    "Mercedes-Benz_CLS-class": "Benz,Benz CLS Class",
    "Mercedes-Benz_E-class": "Benz,Benz E Class",
    "Mercedes-Benz_GLK-class": "Benz,Benz GLK Class",
    "Nissan_Qashqai": "Nissan,Qashqai",
    "Peugeot_508": "Peugeot,Peugeot 508",
    "Porsche_Cayenne": "Porsche,Canyenne",
    "Porsche_Panamera_2010": "Porsche,Panamera",
    "Skoda_Rapid": "Skoda,Rapid",
    "Suzuki_Alto": "Suzuki,Alto",
    "Suzuki_Vitara_2009": "Suzuki,Grand Vitara",
    "Toyota_Yaris_2011": "Toyota,Yaris",
}
