import json
import numpy as np
import os


# RSSI 임계값과 AP 히트율 설정
RSSI_TRHESHOLD = -85 # 최소 RSSI 값
MIN_HIT_RATE = 0.8 # AP 히트율_한 .txt내 해당 히트율 이상의 MAC주소 값만 사용
SSID = ["Smart-CAU_5G", "eduroam", "Smart-CAU_2.4G", "Smart-CAU"] # 고정 MAC주소 만을 이용하기 위해 SSID를 제약

# 데이터 로드 및 전처리 함수
# train folder에 있는 모든 데이터 처리
def load_and_preprocess(folder_path):
    data = []
    file_names = []
    for file_name in sorted(os.listdir(folder_path)):
        file_names.append(file_name)
        with open(os.path.join(folder_path, file_name), "r", encoding='utf-8') as file:
            data.append(json.load(file))
    return data, file_names

############################################

from sklearn.preprocessing import MinMaxScaler

# train data preprocessing
def preprocess_data_with_unified_aps_and_scaling_for_specific_file(raw_data, all_aps, scaling_file_index=None):
    final_data = []

    for i, file_data in enumerate(raw_data):
        file_vectors = []
        for time_slice in file_data:
            time_slice_features = {ap["MAC"]: ap["RSSI"] for ap in time_slice if ap["SSID"] in SSID}
            print(len(time_slice_features))
            vector = [time_slice_features.get(ap, RSSI_TRHESHOLD) for ap in all_aps]
            file_vectors.append(vector)

        file_array = np.array(file_vectors)

        # 특정 파일에 대해서만 정규화 적용
        if scaling_file_index is not None and i == scaling_file_index:
            scaler = MinMaxScaler(feature_range=(0, 1))
            file_array = scaler.fit_transform(file_array)

        final_data.append(file_array)

    #[위치 index][timeslice][mac주소 순서대로 index]
    return final_data

