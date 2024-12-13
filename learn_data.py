import numpy as np
import json
import util
from sklearn.svm import SVC

# 모델(SVM)
class LearnData() :
    def __init__(self) :
        # Train 및 Test 데이터 로드
        train_folder = "train"  # train 폴더 경로
        train_raw_data, self.position_names = util.load_and_preprocess(train_folder)

        # 히트율 + SSID 이름에 따라 mac 주소 필터링
        self.all_aps = []
        for i, file_data in enumerate(train_raw_data):
            hit_features = {}
            for time_slice in file_data:
                hit_features = {ap["MAC"]: hit_features.get(ap["MAC"], 0)+1 for ap in time_slice if ap["SSID"] in util.SSID}
            
            hit_features = {
                mac: value
                for mac, value in hit_features.items()
                if value/len(file_data) >= util.MIN_HIT_RATE
            }
            self.all_aps = sorted(set(hit_features.keys()) | set(self.all_aps))

        train_data = util.preprocess_data_with_unified_aps_and_scaling_for_specific_file(
            train_raw_data, self.all_aps, scaling_file_index=None  # Train 데이터는 정규화 없음
        )

        # Train 데이터 병합
        X_train = [] #RSSI 값에 대한 list
        y_train = [] #true value
        for i, file_data in enumerate(train_data):
            for vector in file_data:
                X_train.append(vector)
                y_train.append(i)
        X_train = np.array(X_train)

        # 모델 학습
        self.model = SVC(kernel='rbf', decision_function_shape='ovr', probability=True)
        self.model.fit(X_train, y_train)  # 학습할 때는 수정된 y_train 사용

    # 클라이언트에게 받은 현재 와이파이 정보를 기반으로, 모델을 돌려보기 위함.
    def detect_position(self, data) :
        wifi_data = json.loads(data)
        preprocess_data = util.preprocess_data_with_unified_aps_and_scaling_for_specific_file(wifi_data, self.all_aps, scaling_file_index=None)
        
        test_data = []
        for file_data in preprocess_data:
            for vector in file_data:
                test_data.append(vector)
        test_data = np.array(test_data)
        test_result = self.model.predict(test_data)[0]
        return f'{self.position_names[test_result]} : {self.model.predict_proba(test_data)[0][test_result]}'
