# CTR_prediction

모델 학습 및 평가를 위해 아래 데이터를 사용합니다.

- [train.csv](https://buzzvil-public.s3.ap-northeast-1.amazonaws.com/ml-assignment/v2/train.csv): 과거 며칠 동안의 버즈빌 광고 노출 데이터 일부입니다. 각 행은 버즈빌 광고가 한 유저에게 노출된 기록을 나타냅니다.
- [test.csv](https://buzzvil-public.s3.ap-northeast-1.amazonaws.com/ml-assignment/v2/test.csv): `train.csv` 파일의 데이터 기간 이후에 수집된 광고 노출 데이터 일부입니다.

데이터는 아래 설명을 참고합니다.

- clicked: 예측해야 하는 타겟 변수입니다. 광고를 클릭했으면 True, 클릭하지 않았으면 False. 
- C1:C26: 26개의 Hashed Categorical Feature. 
- N1:N9: 9개의 Numerical Feature. 
- timestamp: 광고가 노출된 시간.

