## Ramdom Variable( 확률 변수) 
: 일종의 함수로, 확률적인 과정에 따라 값이 결정되는 변수 
- 함수이므로, outcome의 집합을 보면 규칙성을 띄는 경우가 있는데 이를 Probability Distribution(확률분포)라고 함

확률변수는 확률로 표현하기 위한 이벤트를 정의하는 것이다. 
확률이 정의된 sample space에서 확률 변수를 0과 1 사이의 확률로 mapping하는 함수를 확률 함수(확률분포함수)라고 한다. 확률이란 불확실성을 표현하는 수단인데, 이러한 불확실성을 확률로써 계량하기 위해 확률함수를 활용하여 만든 수학적 모형을 확률 모형이라고 한다.


## Sample Space 
: 실험에서 나올 수 있는 모든 가능한 결과(outcome)들의 집합
- Sample Space 위의 모든 확률의 합 = 1


## Event Space 
: 실험에서 잠재적으로 나올 수 있는 결과의 집합
- Sample Space의 부분집합


## Probability 
: Event Space에서 event A가 일어났을 때 P(A)는 사건이 일어날 수 있는 확률에 대한 측정값

<br/>
<br/>

## ⭐️ 확률과 통계(Statistics)의 다른점 
- 확률 : 근본적인 불확실성이 random variable에 의해 포착되고, 어떤 일이 일어났는지 도출하기 위해 확률 규칙을 이용함
- 통계 : 어떤 일이 일어났는지를 관찰하고, 관찰을 설명하는 근본적인 과정, 형태를 알아내려고 노력함

<br/>
<br/>

## Target Space
- X : Q → T 와 같은 함수가 있으면, 이를 Ramdom variable이라고 함
- ex
   - 동전 두개를 던지는 경우를 생각하면, 앞면의 개수를 세는 random variable X가 있으면
   - X(hh) = 2, X(ht) = 1, X(th) = 1, X(tt) = 0
   - Sample Space = [hh, ht, th, tt]
   - 이와 같은 3가지 outcome이 존재함
   - 이경우, T = {0,1,2}
   
### Target Space가 Discrete한 경우
- Random variable X ∈ T로 특정되어 P(X = x)로 표현
- Discrete random variable X에 대한 함수 P = **Probability mass function( pmf )**
- **Joint Probability** : 여러 개의 random variable에 대한 확률분포
  - P(X = x_i, Y = y_j) = P(X = x_i ∩ Y = y_j)
  - **Marginal Probability** : X, Y 각각에 대한 확률 P(x_i), P(y_j)
  - **Conditional Probability(조건부 확률)** : 한 Random Variable을 고정시킨 상태에서 나머지 random variable에 대한 확률 P(y|x)
     - p(x,y) = p(y|x)p(x)
### Target Space가 Continuous한 경우
- Ramdom variable X를 P(a<= X <= b)와 같이 구간으로 표현
- P(X <= x)와 같이 X를 x 이하로 제한하여 표현한 함수 P = **Cumulative distribution fucntion( cdf )**


## Bayes' Theorem
: 관찰된 다른 random variables(y)를 통해 관찰되지 않은 unobserved or latent random variables(x)을 추정
P(x|y) = P(y|x)P(x) / P(y)
← p(x,y) = p(y|x)p(x) = p(x|y)p(y)
- 사전확률 ( P(X) ) : 관찰하지 않은 ramdom bariable x에 대한 주관적인 사전지식을 요약한 것
- y에 대한 x의 우도 ( P(y|x) ) : x와 y가 어떻게 연관되어 있는지 나타냄 
- 사후확률 ( P(x|y) ) = quantity of intersest

## Statistically Independent
p(x,y) = p(x)p(y)
- 다음과 같은 관계를 만족하면 x, y는 independent함
- p(y|x) = p(y)
- p(x|y) = p(x)
- V_(X,Y)[x,y] = V_X[x] + V_Y[y]
- Cov_(X,Y)[x,y] = 0

## Conditional Independent
p(x,y|z) = p(x|z)p(y|z)
- x와 y는 independent
- x와 y는 z에 대해 conditional independent

<br/>
<br/>

-----------------------------------------
<br/>
<br/>

# Discrete Probability Distribution
## Binomial Distribution
### Bernoulli Distribution
: 단 1회의 experiment에 대해 다룸
- X : outcome이 성공 혹은 실패인 경우 

![image](https://user-images.githubusercontent.com/65997635/128626299-c1b82cc5-d1b5-4e73-ad84-c2b294eae798.png)

- Mean, Variance 

![image](https://user-images.githubusercontent.com/65997635/128626594-1ee2b1cd-de15-4c9f-b5c4-89a01d784760.png)

### Binomial Distribution
: n번의 독립적인 experiment에 대해 다룸 
- X : n번 중에 성공 횟수

![image](https://user-images.githubusercontent.com/65997635/128626315-294d03f7-d98a-415b-a4ae-695a2f126443.png)

- Mean, Variance 

![image](https://user-images.githubusercontent.com/65997635/128626604-1392c633-0fc7-42ef-8120-77b5aafe29e9.png)

## Multinomial Distribution
: n번의 독립적인 experiment을 시행할 때, 2가지 이상의 결과에 대해 나오는 경우
- X : i번째 outcome이 나온 횟수 X_i

![image](https://user-images.githubusercontent.com/65997635/128626430-2790a598-9363-418c-8992-c32cf0cbf8ef.png)

## Poisson Distribution
: 단위 시간 또는 단위 공간에 어떤 사건이 몇 번 발생할 것인가를 표현

![image](https://user-images.githubusercontent.com/65997635/128633120-523c82cf-b2a2-46cc-9a3e-8e00f8624f86.png)

![image](https://user-images.githubusercontent.com/65997635/128633131-772fd5bf-7e9c-43a2-b261-c664fe6be93b.png)

- 모수( Population Parameter ) λ : 단위 시간 또는 단위 공간에 어떤 사건의 평균 발생 횟수
- X : 발생 횟수
- Mean, Variance : λ, λ

<br/>
<br/>

-----------------------------------------
<br/>
<br/>

# Continuous Probability Distribution
## Gamma Distribution

![image](https://user-images.githubusercontent.com/65997635/128633416-d830aa04-5300-48d4-99f4-10197cc3ba77.png)

### Gamma Function
: 팩토리얼 함수를 복소수까지 확장한 함수

- x > 0 일 때,

![image](https://user-images.githubusercontent.com/65997635/128633271-3ab8fec9-2118-4f5b-9007-4f8f8610ce55.png)

### Gamma Distribution
: α번째 사건이 일어날 때까지 걸리는 시간에 대한 표현

<br/>

= 총 α번의 사건이 발생할 때까지 걸린 시간에 대한 확률분포

<br/>

( Gamma Function에서 확률변수를 X=x라 할 때, 0 ~ inf까지 적분한 값 = 1 )을 이용하면

![image](https://user-images.githubusercontent.com/65997635/128633688-668c0ec3-516c-487b-800f-00b2639d76d6.png)

![image](https://user-images.githubusercontent.com/65997635/128634136-d62ee264-36d4-4e28-b8c3-583009ab1ed0.png)

⬇️

![image](https://user-images.githubusercontent.com/65997635/128634340-488780c0-515b-4bfc-ae8b-60f363d6cf48.png)

- α : shape parameter
- β : scale parameter
- Mean, Variance  = αβ, αβ^2

## Exponential Distribution
: gamma distribution에서 α = 1로 고정

![image](https://user-images.githubusercontent.com/65997635/128634353-8555415e-e742-4a1a-a627-790d2e8cf563.png)

- Mean, Variance : β, β^2

## Beta Distribution
### Beta Function
![image](https://user-images.githubusercontent.com/65997635/128636492-6c17b9f2-c649-48cd-ac4e-176cb1d52896.png)

### Beta Distribution
: 확률변수가 0에서 1사이 값을 가지고 두 개의 모수(α, β)를 갖는 확률분포함수
- 베타확률분포는 확률변수 자체가 0~1값을 가지기 때문에 확률에 대한 모델링을 하기 좋은 함수

![image](https://user-images.githubusercontent.com/65997635/128636669-e67e784a-a23b-4e64-be58-66da5e196168.png)

![image](https://user-images.githubusercontent.com/65997635/128636548-4f02a03b-46dc-449a-8db6-67b81af11a4b.png)


## Conjugate Prior
: Bayes' Theorem을 이용하여 Likelihood와 Prior로 구한 Posterior의 분포가 Prior의 분포와 같게 나오도록 하는 Prior

- Binomial likelihood + Beta prior = Beta Posterior
- Poisson likelihood + Gamma prior = Gamma Posterior

보통 prior는 exponential family에서 고르는 경우가 많다.
   - exponential family : Bernoulli, binomial, Poisson, Gaussian, Laplace, gamma, beta distribution 

Exponential family를 많이 선택하는 이유는 대부분의 데이터들이 이 모양을 띄고 있기 때문이기도 하며, 

만약 likelihood가 exponential family일 때, prior를 ‘좋은’ exponential family로 선택하게 되면 posterior와 prior가 같은 family에 속하게 되기 때문이다. 

<br/>
<br/>

-----------------------------------------

<br/>
<br/>

# Population( 모집단 )과 Parameters( 모수 )
- Parameters : 평균, 분산, 표준편차, 모비율

## Expected Value

![image](https://user-images.githubusercontent.com/65997635/128471729-88e1edc4-56cc-4eba-aec2-9962f254ec91.png)

### Mean
- Discrete random variable X~P(x) : E[f(x)] = ∑ f(x)P(x)
- Continuous random variable X~P(x) : E[f(x)] = ∫ f(x)P(x) dx

### Median
- Discrete random variable : value들을 정렬시켰을 떄 가운데 있는 값
- Continuous random variable : cdf에서 누적확률이 0.5가 되는 값
- 분포가 어느 방향으로 편향되었는지 판단할 때 주로 사용

### Mode 
- Discrete random variable : 가장 많이 등장하는 value
- Continuous random variable : cdf에서 peak일 때의 값

## Covariance (공분산)
: 두 univariate한 random variable X, Y 사이의 Covariance는 X, Y 각각의 평균과 개별값의 차의 곱의 평균과 같음
- Cov_X,Y[ x, y ] = E_(X,Y)[ (x-E(x) (y-E(y) ]
- Cov[ x, y ] = E[ xy ]-E[ x ]E[ y ]
- **Variance** V_X[ x ] : 자기 자신과의 covariance
- Standard deviation : Variance의 제곱근
- **Covariance Matrix** : multivariate random variable X가 얼마나 퍼져 있는지에 대한 정도
- Covariance > 0 : X가 증가할 때 Y도 증가 ( dependent한 관계 )
- Covariance < 0 : X가 증가하면 Y는 감소 ( dependent한 관계 )
- Covariance = 0 : Independent한 관계
   - 주의 : independent해도 cov ≠ 0 일 수 있음
   - 주의 : convariance = 0이여도 independent한 관계가 아닐 수 있음
      - 예시 : y=x^2(dependent한 관계) → cov[x,y] = E[xy] - E[x]E[y] = 0
- Covariance가 두 변수 X와 Y사이에 어떤 상관관계가 있는지는 알려주지만 상관관계가 얼마나 큰지는 제대로 반영하지 못한다. 
   - 두 random variable의 variance에 영향을 받기 때문에 객관적인 비교가 어려움
   - 따라서, normalize한 **correlation**을 사용

## Correlation (상관관계)
: 두 random variable이 서로 연관성이 있는지 나타냄
- Corr[ x, y ] = Cov[ x, y ] / sqrt( V[ x ]V[ y ] )
- 확률변수의 절대적 크기에 영향을 받지 않도록 공분산을 단위화 시킨 것
- 1또는 -1에 가까울수록 상관성이 크고, 0에 가까울수록 상관성이 없음

<br/>
<br/>

-----------------------------------------
<br/>
<br/>

# 표본 추출
## Central Limit Theorem ( 중심극한정리 )
: 모집단이 평균이 μ이고 표준편차가 σ인 임의의 분포를 이룬다고 할 때, 이 모집단으로부터 추출된 표본의 크기 n이 충분히 크다고 하면, 표본 평균들이 이루는 분포는 평균이 μ이고 표준편차가 ( σ / sqrt(n) )인 정규분포에 근접함

- 심지어 모집단의 모양이 어떻든 관계없이 중심극한 정리는 성립함
- 또한, 표본을 추출하는 모집단이 서로 독립적이라면 여러 모집단에서 추출한 표본이더라도 표본 평균의 분포는 정규분포에 근사함

## 큰 수의 법칙
: 사건을 무한히 반복할 때 일정한 사건이 일어나는 비율은 횟수를 거듭하면 할수록 일정한 값에 가까워지는 법칙

- 어떤 모집단에서 표본들을 추출할 때, 각 표본의 크기가 커지면(시행 횟수가 늘어나면) 상대도수와 모비율의 값이 같아질 확률이 높아진다는 의미

## 표본통계량

![image](https://user-images.githubusercontent.com/65997635/128592047-249a950e-08ca-4c36-8196-de599943b009.png)

: 표본은 매번 추출할 때마다 변하므로 포본 통계량 또한 매번 다름
- 표본 평균, 표본 분산, 표본 표준편차, 표본 비율

## 표준 오차 ( Standard Error of Mean, SEM )
: 표본 평균의 표준편차 = 모집단의 표준편차 / sqrt( 표본의 크기 )
- 표본 통계량의 불확실도로 이해할 수 있음
- 표본 평균의 분포의 너비보다 모집단의 분포의 너비가 더 넓음

![image](https://user-images.githubusercontent.com/65997635/128592679-fb451ad9-b254-49a1-84b4-e72b9542ae5f.png)

![image](https://user-images.githubusercontent.com/65997635/128592692-66041918-bca5-473c-9560-e73487afc58d.png)

![image](https://user-images.githubusercontent.com/65997635/128592733-ca1bcbc0-0e60-4706-956a-2cbf7103e73a.png)
![image](https://user-images.githubusercontent.com/65997635/128592705-7740c4b4-de39-445f-9c50-1c4f26037efd.png)

![image](https://user-images.githubusercontent.com/65997635/128592764-f0e4aa7f-ac52-49a4-95f0-13ed182f5652.png)

- 표본의 크기가 클 수록 두 표본의 평균 값을 더 확실히 계산할 수 있으므로 표본 평균에 대한 표준 오차가 작아짐

### 신뢰구간
표본은 모집단에서 랜덤하게 선택된 것들이기 때문에 각 표본들의 평균을 모아보면 정규분포의 형태를 가짐

![image](https://user-images.githubusercontent.com/65997635/128597077-0c9b81ef-9eb0-4299-9b34-c7280833bd4f.png)

- 정규분포에서 평균값을 중심으로 (2 x 표준편차)의 범위는 약 95%의 면적을 차지함.

   = 모집단에서 뽑은 표본의 평균은 모평균으로부터 ±2 x SEM의 범위 안에 95%의 확률로 들어옴.

   = 100번 정도 샘플링을 했을 때, 모평균이 95번 정도는 (표본들의 평균 ± 2 x SEM) 안에 들어옴

<br/>
<br/>

## Sample Variance ( 표본 분산 )

![image](https://user-images.githubusercontent.com/65997635/128593193-69f192cc-8ff0-45fc-9340-7238efbe8494.png)

- n 대신 n-1로 나누는 이유 : **표본 분산의 기대값 = 모분산**을 만들기 위해서

![image](https://user-images.githubusercontent.com/65997635/128593227-79ede5d8-abc7-40ad-b715-3dfb294e8ddc.png)

![image](https://user-images.githubusercontent.com/65997635/128593243-d055f4be-22ea-421a-8105-ab6cccc018a0.png)

![image](https://user-images.githubusercontent.com/65997635/128593256-e2632162-01c2-4c04-b6fd-7d6e3baaaf67.png)

<br/>
<br/>

## 검정 통계량( test statistic )
: 통계적 가설의 진위 여부를 검정하기 위해 표본으로 부터 계산하는 통계량으로 표본 통계량을 2차 가공한 것
- 통계적 가설의 진위 여부 검정 = 검정통계량의 값이 기준을 벗어나는지 확인하여 세워둔 가설이 틀렸는지 확인하는 과정

### t-value 
: 두 표본 집단 간의 차이를 비교하기 위해 사용하는 지표
- 하지만, 표본의 평균은 항상 **표본 오차( 표본 평균의 불확실도 )** 가 존재함
- 따라서, 표본평균 차이의 통계적 지표 = ( 두 표본 평균의 차이 / 두 표본 평균간 차이의 불확실도 )
![image](https://user-images.githubusercontent.com/65997635/128594764-25b9a78d-81ab-4535-851c-66f0c33c8d08.png)

![image](https://user-images.githubusercontent.com/65997635/128594794-70d58254-afc1-4666-8c8e-7e1a68e20900.png)

![image](https://user-images.githubusercontent.com/65997635/128594798-f69aca99-c21f-43b6-be57-e23bc26a377c.png)


### F-value
: 여러 표본 집단 간의 차이를 비교하기 위해 사용되는 지표
- t-value와 동일하게 여러 표본평균 차이의 통계적 지표 = ( 두 표폰 평균의 차이 정도 / 두 표본 평균간 차이의 불확실도 )를 담고 있음
- 하지만, 차이 정도와 불확실도를 다르게 구하고자 함.

![image](https://user-images.githubusercontent.com/65997635/128595848-b300dfb3-8b5a-42b8-bd63-66186379a0ff.png)

- 표준 오차 : ![image](https://user-images.githubusercontent.com/65997635/128595869-4cad58b3-cbe5-4b90-8b4f-7256cfc5bc9d.png)

- 각 표본 집단 분산 : ![image](https://user-images.githubusercontent.com/65997635/128595900-aa0c8b05-a96a-4a62-9083-0437abd39b4c.png)
→ ![image](https://user-images.githubusercontent.com/65997635/128595902-a01e6b5d-cf2b-4917-a789-1551e841156f.png)

- 전체 표본 분산 : 각 표본 집단 분산의 산술평균

<br/>
<br/>

## 귀무가설 vs 대립가설
- 귀무가설 : 새로울 게 없다
- 대립가설 : 새로운 것이 있다.
- 예시
   - 흡연 여부가 뇌 질환 발생 증가에 영향을 미치는지 연구한다고 하면,
   - 귀무가설: 흡연 여부는 뇌혈관 질환의 발생에 영향을 미치지 않는다.
   - 대립가설: 흡연 여부는 뇌혈관 질환의 발생에 영향을 미친다.

- 귀무가설을 사용하는 이유
1. 참이 아님을 증명하는 것이 참임을 증명하는 것보다 훨씬 쉽우므로
2. 귀무가설을 “올바르게” 서술하는 것이 대립가설을 “정확하게” 서술하는 것 보다 실패할 가능성이 적으므로


## P-value
: 귀무가설이 맞다는 전제 하에, 귀무가설이 말이 될 확률
- 1종 오류( 실제로는 일이 일어나지 않았는데도 기각해버린 것 )를 범할 확률

가령 우리가 두 표본 집단의 특징값의 평균이 통계적으로 유의한 차이가 있는지 검증할 때, 두 표본 집단으로부터 검정 통계량(ex. t-value)을 계산할 수 있음. p-value는 이 검정 통계량에 관한 확률인데, 우리가 얻은 검정 통계량보다 크거나 같은 값을 얻을 수 있을 확률을 의미함
-  p-value가 5%보다 작으면 유의한 차이가 있다고 얘기함

- t-value는 표본 수(즉, 자유도)에 따라 모양이 다르다보니 같은 t-value라고 하더라도 표본 수에 따라서 표본 간의 차이가 충분히 크다고 할 수도 있고 그렇지 않다고도 할 수 있음
- 따라서, p-value는 표본 수와 검정통계량의 분포 모양에 상관없는 확률로 정의
 
<br/>
<br/>

-----------------------------------------

<br/>
<br/>


