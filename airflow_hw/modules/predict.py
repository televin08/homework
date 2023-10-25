import os
import dill
import pandas as pd
import json

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
def predict():
    files_directory_pkl = f'{path}/data/models'
    files_list_pkl = os.listdir(files_directory_pkl)
    filename = f'{path}/data/models/{files_list_pkl[1]}'
    files_directory = f'{path}/data/test'
    files_list = os.listdir(files_directory)

    with open(filename, 'rb') as file:
        modelData = dill.load(file)

    id_list = []
    predict_list = []
    for name in files_list:
        with open(files_directory + '/' + name, "r") as json_file:
            json_data = json.load(json_file)
            df = pd.DataFrame.from_dict([json_data])
            y = modelData.predict(df)
            id_list.append(json_data['id'])
            predict_list.append(y[0])

    predic_dict = {'id': id_list, 'predict': predict_list}
    predictions_df = pd.DataFrame(predic_dict)
    predictions_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

if __name__ == '__main__':
    predict()