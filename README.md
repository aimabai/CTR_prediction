# CTR_prediction

모델 학습 및 평가를 위해 아래 데이터를 사용합니다.

- [train.csv](https://buzzvil-public.s3.ap-northeast-1.amazonaws.com/ml-assignment/v2/train.csv): 과거 며칠 동안의 버즈빌 광고 노출 데이터 일부입니다. 각 행은 버즈빌 광고가 한 유저에게 노출된 기록을 나타냅니다.
- [test.csv](https://buzzvil-public.s3.ap-northeast-1.amazonaws.com/ml-assignment/v2/test.csv): `train.csv` 파일의 데이터 기간 이후에 수집된 광고 노출 데이터 일부입니다.

데이터는 아래 설명을 참고합니다.

- clicked: 예측해야 하는 타겟 변수입니다. 광고를 클릭했으면 True, 클릭하지 않았으면 False. 
- C1:C26: 26개의 Hashed Categorical Feature. 
- N1:N9: 9개의 Numerical Feature. 
- timestamp: 광고가 노출된 시간.

제공한 데이터를 활용하여 모델을 개발하고, 아래 두 파일을 작성하여 제출합니다.

- `MODEL.md`: 데이터 전처리 과정, 모델 선정 방법, 결과 지표 등을 간략하게 설명하는 파일을 작성합니다.
  - 본 과제 수행 과정에서 LLM(ChatGPT, Gemini 등)을 활용해도 무방합니다.
  - LLM을 활용한 경우, `MODEL.md` 마지막에 [부록] 섹션을 두고 어떤 단계에서 어떤 목적으로 사용했는지를 **필수로** 기재해 주세요.
- `assignment.py`: 모델 구현 등의 코드를 작성합니다. Python 외 다른 언어도 사용 가능하며 언어에 맞는 확장자를 사용해주시면 됩니다.

> Jupyter Notebook 파일도 허용합니다. 이 경우 `MODEL.md`에 기록할 내용을 Notebook 파일 Markdown Cell에 기록하여 Notebook 파일만 제출하셔도 됩니다. 

## 제출 기한
제출 기한은 3일이며 필요한 경우 담당자와 조율할 수 있습니다.

## 과제 제출 방법
`과제 가이드`의 내용을 구현하여 github main branch에 반영한 뒤, 제출 완료 이메일을 버즈빌 채용 담당자에게 보냅니다.
