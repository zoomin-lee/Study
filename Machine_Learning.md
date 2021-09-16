# MLE and MAP

- Deep Learning : Data를 가장 잘 설명하도록 parameter들을 구하는 것
- Deep Learning의 기본적인 Loss Function들은 대부분 Maximum Likelihood Estimation( MLE )과 Maximum A Posterior( MAP )를 통해 증명 가능함

<br/>

문제상황 정의
- y(x|w) : parameter w를 가지고 키(x)를 넣으면 예측한 몸무게(t)가 나오는 함수
- parameter w를 잘 학습하여 y(x|w)를 얻으면 우리는 모든 가능한 키(x)에 대해여 실제 몸무게(t)를 예측할 수 있음
- t = y(x|w) 과 같이 항상 일치한다고 말할 수 없음
- 즉, 실제 몸무게(t)는 예측한 몸무게(y)일 확률이 가장 높지만, 아닐 수 있음

## Maximum Likelihood Estimation( MLE, 최대우도법 )

- 따라서, random variable t는 예측한 몸무게(y)를 평균으로 하는 Gaussian Distribution를 따른다고 볼 수 있음

![image](https://user-images.githubusercontent.com/65997635/128605496-e9c6a8a4-93a4-4b66-b803-ddf1e8d145ac.png)

- σ : 예측이 얼마나 불확실한지 나타내는 정도로 문제의 특성에 따라 설정되는 상수 값
   - σ이 작다는 것은 t 값를 더 확신한다는 뜻 
   - 반대로 σ이 크다는 것은 t 값에 자신이 없다는 뜻

![image](https://user-images.githubusercontent.com/65997635/128605598-1c2bcc73-1cd5-45d3-8b64-a363345fac5d.png)

- p(t|X) : input x일 때 실제 몸무게가 t일 확률 밀도

### Likelihood 

![image](https://user-images.githubusercontent.com/65997635/128605790-b3ff5854-d7b1-42a0-be8b-f9e1b64e5547.png)

- p(D|w) : dataset D에 대해여 키가 x_1일 때 실제몸무게가 t_1이고, 키가 x_2일 때 실제몸무게가 t_2이고, ...일 확률
   - data들이 모두 독립이므로 곱의 법칙을 통해 다음과 같이 구함 
- 따라서, p(D|w)가 최대가 되는 w를 찾아야함 = MLE

![image](https://user-images.githubusercontent.com/65997635/128605873-d81679cc-b7b8-444e-b15d-a6c9ab6e9a4a.png)

σ와 π는 상수 값이므로 log likelihood를 최대화 시키는 w에 영향을 주지 못하므로 제거하면

![image](https://user-images.githubusercontent.com/65997635/128605926-581f95b4-a21c-4cd5-87b5-9d43877627f5.png)

regression 시 가장 많이 이용하는 loss function인 MSE(Mean Squared Error) 식이 구해짐

- 즉, L2 loss를 최소화 시키는 것 = Likelihood를 최대화 시키는 것
- Classification 문제 : Gaussian Distribution 대신에 Bernoulli Distribution을 이용하면 Cross Entropy Error 유도 가능
   - Bernoulli Distribution : p(x) = p^x * (1-p)^(1-x) 

## Maximum A Posterior ( MAP )
Likelihood와 Posterior의 차이 : Prior의 유무로 Posterior는 Prior가 포함되어 있음
- MLE : 데이터만을 이용해서 구하고 싶음
- MAP : 데이터와 사전지식을 반영하여 구하고 싶음

![image](https://user-images.githubusercontent.com/65997635/128606274-05b96f40-1d16-4efc-8843-24ed5ac4d244.png)

![image](https://user-images.githubusercontent.com/65997635/128606353-d7168c38-b18e-470c-8930-fb95dd11d5a0.png)

![image](https://user-images.githubusercontent.com/65997635/128606357-12b6ebef-7c97-4f57-b8e1-dbc8b98b73aa.png)

### Prior P(w)
- prior를 가지고 있다면, w값을 구하는데 있어 도움이 됨

BUT 별다른 Prior가 없더라도 Prior를 반영하는 것이 좋은 경우가 많음

- output에 대한 특정 제약조건을 걸고 싶은 경우
   - 예시 : overfitting을 방지하고자 w 값이 0에 가깝도록 제한을 주기 위하여 w가 0을 평균으로 하는 Gaussian Distribution이라는 prior를 걸어줄 것임

![image](https://user-images.githubusercontent.com/65997635/128606365-99cb46e1-72f4-4727-8117-0e4959f8ec86.png)

![image](https://user-images.githubusercontent.com/65997635/128606439-8071a55e-8769-4f59-a199-190f9f2533c0.png)

log p(D|w)는 likelihood 이므로 이 값을 maximize하는 것은 아래의 L(w)을 minimize 하는 것과 같음

![image](https://user-images.githubusercontent.com/65997635/128606545-ddf4d5a6-c203-4254-869e-918e86fc091d.png)

![image](https://user-images.githubusercontent.com/65997635/128606626-4bc3f14f-9852-4669-ae23-6cb217834596.png)

여기서 상수 항을 모두 제거하면 다음 식을 minimize하는 문제가 됨

![image](https://user-images.githubusercontent.com/65997635/128606642-533d2917-9c07-4564-9c21-00c63700bdb6.png)

- Weight Decay(L2 Regularization) 방식을 적용한 Deep Learning의 Loss 함수를 유도함
- L1 Regularization : Laplacian Distribution을 Prior로 걸어주면 유도 가능

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Bias and Variance

## Bias
: 실제 모델의 파라미터 **θ**와 데이터셋을 통해 추정한 θ의 차이
- bias(θ) = E(θ) - **θ** = E(θ - **θ**)
- unbiased : bias = 0

## Variance 
: θ의 평균과 θ들간의 차이 즉, θ가 흩어진 정도 

![image](https://user-images.githubusercontent.com/65997635/128623241-e71d1fd9-c39e-4e1c-afdf-0d41253ea5cf.png)

train data 또는 test data에 대한 결과를 bias와 variance 관점에서 해석한 fig

<br/>

- (a) 
- (b) bias가 작고, variance가 큰 모델
- (c) bias가 크고, variance가 작은 모델
- (d) bias, variance 둘 다 큰 모델

<br/>

train data에서는 (a)와 같은 결과였는데 test data를 넣어보니 (b), (c), (d)와 같은 결과가 나왔다면 loss가 클 것임

<br/>

하지만, 모두 같은 원인으로 loss가 큰 것은 아님

<br/>

(b) overfitting 
- train data와 test data의 차이 = variance의 차이
- train data와 test data는 모집단의 샘플링을 통해 만들텐데, 이런 표본 집단은 표준 오차가 발생함
- 따라서, train data에 overfit하게 학습하면 이런 표준 오차까지 학습하게 되므로 test data에는 맞지 않게 됨 

(c) train data에 대한 결과라면 underfitting / test data에 대한 결과라면 train data와 test data 간의 차이
- train data와 test data의 차이 = 평균의 차이

(d) (b)와 (c)의 원인 동시에

<br/>

## Mean Square Error와 bias, variance 관계

![image](https://user-images.githubusercontent.com/65997635/128625108-a04022c1-6000-462c-9515-222208f85d5d.png)

: E[(y - f(θ)] = E[(f(**θ**) - f(θ)] =  E[f(θ) - E(f(θ))^2] + (E[θ] - **θ**)^2 = Var(θ) + Bias(θ, **θ**)^2

- MSE = Variance + Bias^2

1. (b) train data에 잘 맞게 학습시키는 것 = 모델 복잡도를 높이는 것 

→ test data에서 variance가 커져 total loss는 오히려 증가함

2. (c) 모델 복잡도를 단순하게 함 = 학습이 부족함 

→ test data에서 bias가 커져 total loss는 오히려 증가함

- 즉, bias와 variance는 trade off 관계임


## Bias Variance Trade off
: train data 학습 시, MSE의 최소값을 찾으면 bias와 variance가 거의 같을 때 최적값이 될 것임

- 따라서, train data들의 b-v trade off가 아니라 train 후 test 할 때의 b-v trade off를 고려해야함
- train 에서 MSE의 최적 점을 찾았는데 train set과 test set의 차이로 test set에서는 이것이 최적점이 아닐 수 있다는 것
- 단, train error가 커서 train error에 대한 분석을 할 때는 train의 b-v에 대해서 고민 필요

## Deep Learning에서 Bias Variance Trade off
: Deep Learning에서도 b-v trade off를 피하기 위해 Big 데이터가 전제가 되어야 함
- 큰수의 법칙을 통해 데이터가 많으면 train data가 모집단과 비슷해지므로  모델 복잡도를 높일 수 있음 
- 하지만, 모든 경우에 Big 데이터가 있는 것은 아니여서 딥러닝에서도 이러한 문제를 없애기 위한 다양한 trick ( regularization, dropout, domain adaptation ...)이 존재함

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Cross Entropy

## Information Gain ( 정보량 )
- 확률이 낮은 사건(= 깜짝 놀랄만한 정도 )일수록 정보량이 높음
- 어떤 사건의 random variable X에 대한 정보량 ∝ 1/P(X)

![image](https://user-images.githubusercontent.com/65997635/128668966-59d402b3-ee62-4bc6-b425-f7486e6b6ffa.png)

## Entropy ( 평균 정보량 )

![image](https://user-images.githubusercontent.com/65997635/128669073-295f6f17-f6c7-4e60-987e-a5ae8f5576a5.png)

= 각 label의 확률분포 
- 엔트로피는 불확실성의 척도로 높으면 정보가 많고, 확률이 낮다는 것
- 즉, 어떤 데이터가 나올지 예측하기 어려운 경우

## Cross Entropy
: 예측과 달라서( 실제값과 예측값이 다름 정도에 따라 ) 생기는 정보량
- 실제 분포 P(x)에 대하여 알지 못하는 상태에서, 모델링을 통하여 구한 분포인 Q(x)를 통하여 P(x)를 예측하는 것


- Q(x) : 예측값
- P(x) : 실제값

### Binary Cross Entropy

![image](https://user-images.githubusercontent.com/65997635/128669310-57cebacc-2a85-4201-b851-d8d72174e1d6.png)


### Cross Entropy

![image](https://user-images.githubusercontent.com/65997635/128669422-1c99a8bc-a0e9-4618-9e8c-0e7e95bd8fad.png)


## KL Divergence (= Relative Entropy )
: P(x)와 Q(x)의 차 

![image](https://user-images.githubusercontent.com/65997635/128671346-feecf17e-67f6-4866-aff4-88b196bce1da.png)

- Q(x)를 P(x)에 가깝게 하는 것이 목표이므로 KL Divergence를 줄여야함
- 이때, H(P)는 상수값이므로 사실 KL Divergence를 줄이는 것 = Cross Entropy를 줄이는 것
- 따라서, deep learning의 손실 함수로 Cross Entropy를 사용

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Support Vector Machine ( SVM )

## 라그랑주 승수법
: 최적화 문제에서 최대/최소값을 찾으려는 문제에서 해결방법으로 사용

- 목적 함수 f(x, y)
- 제약 조건 g(x, y) = 0
- 새로운 변수 λ

![image](https://user-images.githubusercontent.com/65997635/128685277-882db68f-a000-4e7c-b734-d2cbdd76e3a2.png)

다음의 보조방정식에 대해 모든 변수에 대한 편미분 값이 0이 되는 변수의 해를 찾는 것

- WHY ? 제약 조건을 만족시키면서 목적 함수를 최대/최소화 시키는 점에서는 목적함수의 gradient와 제약 조건의 gradient가 평행하기 때문

### KKT Condition
: 부등식의 제약 조건에서도 쓸 수 있게 확장시킨 것

## Decision Rule
: 새로운 입력의 class가 + 인지 - 인지 결정하는 방법

![image](https://user-images.githubusercontent.com/65997635/128685830-44e7ecb9-3d0c-4d34-80a9-86362c95593f.png)

- **w** : Decision Boundary의 법선 벡터
- **u** : 새로운 input
- Decision Boundary ( hyperplane ) 
   - Decision Boundary 위에 존재하는 negative/positive sample에 대해 같은 margin( δ )만큼 떨어져 있음  
   ![image](https://user-images.githubusercontent.com/65997635/128686428-4dbf517c-4c0f-497b-adb2-669d035e3428.png)
   - 하지만, w와 b 모두 임의의 수이므로 양변을 δ로 나누어 우변을 1로 나누어도 상관없음
   ![image](https://user-images.githubusercontent.com/65997635/128686440-fb087cb5-cfdc-4c23-9397-0b12523cf512.png)
   
![image](https://user-images.githubusercontent.com/65997635/128686679-549f4547-8415-4d31-9354-e969990637d8.png)

![image](https://user-images.githubusercontent.com/65997635/128687254-58551782-c4b0-4503-9449-dc21ec850ae7.png)

## Margin을 최대화

![image](https://user-images.githubusercontent.com/65997635/128686888-d074a874-c1c1-4341-b3fc-977f5cbce40a.png)

두 점 사이의 거리 
: ![image](https://user-images.githubusercontent.com/65997635/128687684-6640d315-ea6f-4430-8e32-e06aa0b81b75.png)

우리는 이를 최대화 하고 싶음
max 1/|w| = min |w| = min |w|^2

## 라그랑주 승수법과 Decision Boundary

![image](https://user-images.githubusercontent.com/65997635/128688582-f918a9b2-ace3-4221-ad97-ccb9f890782c.png)

- L의 최대값은 결국 x_i와 x_j에 의해 결정되는 것을 알 수 있음 
- 즉, 이를 잘 변형시켜 준다면 L의 최대값을 더 끌어올려줄 수 있음

## Kernel 

![image](https://user-images.githubusercontent.com/65997635/128689073-adf4c242-6701-49a8-af99-33fa415fa744.png)

또한, 선형적으로 해결할 수 없는 문제는 kernel을 사용하면 데이터를 고차원에 mapping 시켜 해결할 수 있음

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Naive Bayes 

## 판단 근거
사전지식( prior ) x likelihood( 추가 정보 ) 

- 175cm인 사람이 남자일 확률 : P(성별 = 남자) x P(키 = 175cm | 성별 = 남자)
- 175cm인 사람이 여자일 확률 : P(성별 = 여자) x P(키 = 175cm | 성별 = 여자)

<br/>

만약, 몸무게 정보가 추가된다면?(= feature 가 늘어난다면? )
- 175cm, 80kg인 사람이 남자일 확률 : P(성별 = 남자) x P(키 = 175cm | 성별 = 남자) x P(몸무게 = 80kg | 성별 = 남자)
- 175cm, 80kg인 사람이 여자일 확률 : P(성별 = 여자) x P(키 = 175cm | 성별 = 여자) x P(몸무게 = 80kg | 성별 = 여자)

<br/>

## 판단 근거 구하는 법
: 위와 같은 예시처럼 판단 근거를 구하는 이유는 다음과 같음

- 주어진 데이터 x에 대해서 class i라고 판단할 확률

![image](https://user-images.githubusercontent.com/65997635/128715176-069a61f0-17e6-4fe8-9e4f-fd15e4bc1d62.png)

- x의 feature가 여러 개라면?

![image](https://user-images.githubusercontent.com/65997635/128715437-f25d22e6-3fbd-49c5-9f4d-a162ea86fff2.png)


이때, feature들이 서로 독립적이라고 가정한다면 

![image](https://user-images.githubusercontent.com/65997635/128715543-18ada221-2295-4a29-8f6d-f24f5659aeeb.png)

이와 같이 구할 수 있음

## 나이브 베이즈 분류기

![image](https://user-images.githubusercontent.com/65997635/128716878-3d32ffa7-9eed-4999-882c-aa626b188f7a.png)

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# ROC Curve
## Confusion Matrix

|       | Positive       | Negative       |
|-------|----------------|----------------|
| True  | True Positive  | False Positive |
| False | False Negative | True Negative  |

- Accuracy : TP / ( TP + FP + FN + TN )
- Precision : TP / ( TP + FP )
- Recall : TP / ( TP + FN )
- F1 Score : ( 2 x Precision x Recall ) / ( Precision + Recall )

## TPR과 FRP의 관계

![image](https://user-images.githubusercontent.com/65997635/128721596-786b0b86-eec7-476c-ae79-44af16e05fbb.png)

- Threshold가 낮으면, TPR과 FPR 동시에 높아짐
- Threshold가 높으면, TPR과 FPR 동시에 낮아짐

## ROC Curve 분석
- 현재 이진 분류기의 분류 성능은 변하지 않되, threshold 별 FPR과 TPR의 비율을 Plot한 그래프
   - plot 위의 한 점 : 해당 threshold일 때의 FPR과 TPR의 비율 

![image](https://user-images.githubusercontent.com/65997635/128722347-1e91f605-7ced-4be3-8fd2-efaf0b42755d.png)

![image](https://user-images.githubusercontent.com/65997635/128722364-040391d4-d6f5-4099-bc30-13148098607e.png)

- class를 더 잘 구별하면 ROC curve는 좌상단에 더 가까워짐

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Autoencoder
: 고차원 data의 manifold를 찾고, 이를 encoder가 latent vector로 표현하고, decoder가 latent vector를 복원하여 고차원 manifold 위에 위치하게 만드는 것

![image](https://user-images.githubusercontent.com/65997635/128733256-852dafe6-f8b7-4686-aa07-8326a8c7c261.png)

## Manifold

![image](https://user-images.githubusercontent.com/65997635/128734405-a20d3fc6-7cf7-4320-9249-4de63ab3c17b.png)

다음 3차원에 표현된 manifold를 encoder를 거치면 2차원 평면으로 표현 가능함

- input이 manifold에서 조금 떨어져 있으면 정보가 손실되어 manifold 위에 reconstruct image가 생김

## Linear Autoencoder : PCA

## Nonlinear Autoencoder 

![image](https://user-images.githubusercontent.com/65997635/128735350-d5422c19-151e-4841-af41-1f001a9d2ee4.png)

< MNIST 데이터의 Autoencoder 거친 후 latent vector 시각화 >

한계점
1. 보지 못했던 데이터를 생성할 때 퀄리티가 떨어짐
   - latent space에서 빈 공간에서부터 생성된 데이터들은 품질이 떨어짐
   - 이는, encoder가 특정 데이터를 인코딩 해놓은 좌표에서만 잘 작동하는 overfitting이라고 볼 수 있음
2. encoder 시 latent vecotr들의 위치 선택에 특정한 규칙이 없음
   - decoding 시 encoder가 latent space의 어떤 좌표에 어떤 데이터를 옮겨 담았는지 매번 sampling하여 확인해야함
3. 어떤 class는 매우 작은 영역에 밀집되지만, 어떤 class는 넓게 퍼질 수 있음 

# Variational Autoencoder ( VAE )
: Autoencoder의 한계점을 보완

1. encoder가 latent space에 데이터를 encoding할 때 gausssian 분포를 이용하여 randomness를 추가해줌

![image](https://user-images.githubusercontent.com/65997635/128816624-3fc99446-3623-4b7a-8db6-56a89a11a5cf.png)

- Gaussian 분포의 평균과 분산은 Neural Net이 계산함
- 본 정규분포로부터 샘플링된 값에 encoding 함
- 결국, label들의 분포는 정규 분포의 형태로 클러스터링 됨

2. encoder가 embedding시키는 latent space의 형태에 대해 평균이 0이고 표준편차가 1인 표준 정규분포의 형태로 제약조건을 검
- 추후 decoding을 수행할 떄 선정할 latent space의 임의의 좌표를 쉽게 sampling하기 위함
- KL Divergence를 사용함

<br/>

⬇️

<br/>

1, 2번을 통해 각각의 label들은 클러스터를 이루면서도 동시에 전체적인 embedding 분포는 표준정규분포를 따르게 됨
- 이 과정에서, 각각의 label에 해당되는 클러스터들은 최대한 비슷한 variance를 갖고 분포하도록 encoder가 학습하게 됨
- 따라서, 3번 해결

![image](https://user-images.githubusercontent.com/65997635/128818094-a8a50376-f5c4-48b3-84c9-1d863ff7280b.png)

< MNIST 데이터의 Variantional Autoencoder 거친 후 latent vector 시각화 >

<br/>

## 수식을 통한 이해

## Decoding
- latent variable z를 사용하여 새로운 P(x) 정의

![image](https://user-images.githubusercontent.com/65997635/128838387-39c017cc-7c23-430b-a1b5-d2db6067c2b2.png)

- 하지만, 이 수식은 모든 z에 대한 P(x|z)을 계산할 수 없음

### P(z) : latent variable의 prior distribution
: gaussian distribution
- z는 각 차원이 독립이므로 해석 가능한 요소들이 encoding 된거임
   - 예시 : 사람 얼굴 이미지를 생성할 때, z에는 웃음의 정도, 눈썹 위치 등의 feature가 있을 수 있고, 이런 feature 들은 각각 정규분포를 따르고 있음  

### P(x|z)
: 베이즈 정리를 통해 분해해보아도, P(x) 때문에 구할 수 없음

![image](https://user-images.githubusercontent.com/65997635/128839207-a1cc50de-0057-4ddc-a963-46f96355bbde.png)

- P(z|x)를 encoder를 통해 근사해야함

## Training

![image](https://user-images.githubusercontent.com/65997635/128859685-4e4edcc3-6ce5-4b0b-9905-6ca00782c610.png)

- θ : decoder parameter
- Φ : encoder parameter

![image](https://user-images.githubusercontent.com/65997635/128859899-4dcbda04-5a60-412e-ba9f-d88404f4b3e2.png)

: decoder network의 출력으로 input data가 얼마나 잘 reconstruct 되었는지 의미함

![image](https://user-images.githubusercontent.com/65997635/128860310-a1707e72-9dee-4d80-a4a8-6dd0309bc1b0.png)

: P(z)와 P(z|x) 사이의 유사도를 의미
- P(z) : gaussian distribution
- P(z|x) : 공분산 행렬을 가정한 gaussian distribution

![image](https://user-images.githubusercontent.com/65997635/128860121-f9a55669-c63a-4643-87a5-cdfb9fad4296.png)

: 이 항은 여전히 구할 수 없지만, KL term 이므로 항상 0보다 크거나 같음 
- 첫번째, 두번째 항만 이용해서 lower bound를 구함

## ❓ Distribution 이란
: 크기가 256x256x3인 사람 얼굴 사진이면 데이터 x는 이만한 크기의 벡터임. 이 사진을 어떤 feature 기준에 따라 일렬로 세웠을 때의 분포를 뜻함
- 예를 들어, 우리가 가지고 있는 사진 데이터셋이 동양인 중심이면 머리색을 기준으로 일렬로 나열했을 때 검은색 머리를 한 부분의 x값의 y값이 커질 것이고, 노란색 머리를 한 부분의 x값의 y값이 작아질 것임. 이를 표현하는 확률 분포를 말함

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Ensemble Method

## Ensemble
: 데이터를 각각 학습한 여러 모델의 예측 결과를 평균하여 예측함

## Bagging(= Bootstrap Aggregating )
: Variance를 줄이는 방향

### Bootstrapping 
: N개의 sample data로 1000개의 bootstrap samples를 만들려면, 복원 추출을 N번 실행하여 새로운 sample data set을 만드는 과정을 1000번 반복함
- 원래 sample data는 모집단의 표본임 = Estimated Population
- Estimated Population에서 복원 추출을 하면 Population과 비슷한 분포를 가짐
- 따라서, data의 variance를 추정하는데 유용함
- OOB( Out-Of-Bag ) Estimator : 복원 추출을 하다보니 한번도 안 뽑히는 데이터가 전체 데이터의 36.8% 생김 

### Bagging
: 모델 별로 bootstapping을 하여 만든 데이터셋을 각각 학습시킨 후 평균 모델을 만듦

- 각 모델 별로 보면 학습데이터에 overfitting되어 low bias, high variance를 가짐
- 이 모델들의 평균을 사용(ensenble)하면 low bias, low variance인 모델이 만들어짐
- tree correlation 이슈 : 특정 feature의 영향력이 크면 독립된 tree여도 비슷한 결과가 나옴

### Random Forest
: 데이터 셋에서 bootstrapping을 하는 것이 아니라 feature로부터 샘플링을 함
- 모든 모델이 서로 다른 feature을 학습함

## Boosting
: Bias를 줄이는 방향
- 맞추지 못한 data에 가중치를 주어 다음 모델이 학습할 데이터 sampling 시 해당 데이터가 뽑힐 확률이 높아지도록 함






