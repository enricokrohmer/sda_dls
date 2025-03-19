# List of models from ssb that are summarised as one 
# single class. Name on the left is the class name which
# is used in the dataset
SIMILAR_MODELS_MODEL = [
    ['Peugeot_508', 'Peugeot_508SW'],
    ['Porsche_Panamera_2010', 'Porsche_Panamera_2016'],
    ['Renault_Megane', 'Renault_MeganeCabrio'],
    ['Skoda_Rapid', 'Skoda_RapidSpaceback'],
    ['Toyota_Prius', 'Toyota_PriusPlug-In'],
    ['Toyota_Yaris_2009', 'Toyota_Yaris_2011'],
]

# Vehicle Models in both datasets which have
# the same model generation and facelift
MAPPING_MODEL_YEAR_FACELIFT = {
    'Audi,Audi A1': 'Audi_A1',
    'Audi,Audi A7': 'Audi_S7',
    'Audi,Audi Q3': 'Audi_Q3',
    'Dodge,Caliber': 'Dodge_Caliber',
    'Jeep,Grand Cherokee': 'Jeep_Cherokee',
    'LAND-ROVER,Range Rover': 'LandRover_RangeRoverSport',
    'MAZDA,Mazda CX-5': 'Mazda_CX-5',
    'Benz,Benz CLS Class': 'Mercedes-Benz_CLS-class',
    'Peugeot,Peugeot 508': 'Peugeot_508',
    'Porsche,Canyenne': 'Porsche_Cayenne',
    'Porsche,Panamera': 'Porsche_Panamera_2010',
    'Skoda,Rapid': 'Skoda_Rapid',
}

# Vehicle Models in both datasets which have
# the same model generation and same generation
# Used in the paper
MAPPING_MODEL_YEAR = {
    'Audi,Audi A1': 'Audi_A1',
    'Audi,Audi A7': 'Audi_S7',
    'Audi,Audi Q3': 'Audi_Q3',
    'Dodge,Caliber': 'Dodge_Caliber',
    'Hyundai ,Santafe': 'Hyundai_SantaFe',
    'Jaguar,Gaguar XF': 'Jaguar_XF',
    'Jeep,Grand Cherokee': 'Jeep_Cherokee',
    'LAND-ROVER,Range Rover': 'LandRover_RangeRoverSport',
    'Lexus,Lexus CT': 'Lexus_CT',
    'MAZDA,Mazda CX-5': 'Mazda_CX-5',
    'Benz,Benz CLS Class': 'Mercedes-Benz_CLS-class',
    'Benz,Benz E Class': 'Mercedes-Benz_E-class',
    'Benz,Benz GLK Class': 'Mercedes-Benz_GLK-class',
    'Nissan,Qashqai': 'Nissan_Qashqai',
    'Peugeot,Peugeot 508': 'Peugeot_508',
    'Porsche,Canyenne': 'Porsche_Cayenne',
    'Porsche,Panamera': 'Porsche_Panamera_2010',
    'Skoda,Rapid': 'Skoda_Rapid',
    'Suzuki,Alto': 'Suzuki_Alto',
    'Suzuki,Grand Vitara': 'Suzuki_Vitara_2009',
    'Toyota,Yaris': 'Toyota_Yaris_2011',
}

