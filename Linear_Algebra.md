## Vector Space ( 벡터 공간 )
: 벡터를 원소로 하는 집합으로, 상수배와 덧셈이 정의되어야 함

## Subspace ( 부분 공간 )
: 벡터 공간내의 부분 집합

- 예시 : 2차원 실수 공간 내의 1차원 부분공간

![image](https://user-images.githubusercontent.com/65997635/128677111-9c0e97c7-cd95-4d26-b121-ef5229db3b88.png)

- 행렬 A의 행/열들의 선형결합으로 구성된 벡터공간(span)은 부분공간임

![image](https://user-images.githubusercontent.com/65997635/128677724-24a3b9a2-f9f0-468e-9340-f83c46c41371.png)

- Row Space ( 행공간 ) : 행렬 A 행의 선형 결합으로 이루어진 선 상에 있는 모든 벡터들의 집합
- Column Space ( 열공간 ) : 행렬 A 열의 선형 결합으로 이루어진 선 상에 있는 모든 벡터들의 집합
- Null Space ( 영공간 ) : Ax = 0을 만족하는 x의 집합
  - Row Space에 직교함

- 행렬 A를 함수라 생각하면
  - 정의역 : Row Space와 Null Space 상의 벡터들의 선형조합
  - 치역 : Column Space
  - 공역 : Column Space + left null space

<br/>

## Linear Independent
1. vector들을 행렬 A의 column으로 넣고 행렬 A의 Null space를 체크
2. 행렬 A의 Null space(Ax = 0)가 오직 zero vector만 존재할 경우, 이 벡터들은 독립임
3. 이때, 행렬 A의 Rank는 coulmn의 수와 같음
4. 따라서 free variable은 존재하지 않음

## Span
vector들을 linear combination했을 때 형성되는 공간
- 사용하는 벡터에 따라서 모든 공간을 채울 수 있고, subspace만 채울 수 도 있음

## Basis
: independent한 vector들이 어떤 공간을 span하는 것 
- 만약 n차원의 공간에 대해 n개의 벡터를 가지고 있을 때, 이들이 basis vector이기 위해서는 nxn 행렬이 역행렬이 존재해야함
- n차원 공간에 대해 무수히 많은 basis가 존재함 

## Rank
: 행렬 A가 나타낼 수 있는 벡터 공간(Column space / Row space)에서 기저의 개수
- Full rank : 행렬 A가 가질 수 있는 최대로 가능한 rank

## 차원( Dimension ) 
- 예를들어, 3차원 공간의 기저가 되기 위해선 반드시 3개의 벡터가 필요함
- 행렬 A의 Rank는 행렬 A의 Column space의 차원임
- 행렬 A의 Null space의 차원은 전체 colum의 개수 - Rank

<br/>

## Covariance Matrix

![image](https://user-images.githubusercontent.com/65997635/128820230-ff111dd1-6ea7-4861-9f60-6cb6e4405e3a.png)

: Covariance Matrix의 고유벡터, 고유값를 구하면, 데이터가 어느 방향으로 얼만큼 퍼져있는지 알 수 있음

<br/>

## Jacobian Matrix
: 미소 영역에서 '비선형 변환'을 '선형 변환'으로 근사 시키는 것

![image](https://user-images.githubusercontent.com/65997635/128667055-3dd1c8c8-6c57-451b-8012-d269a146967e.png)

### Chain Rule
z = f(x, y)에 대해 x = g(t), y = h(t)이면

![image](https://user-images.githubusercontent.com/65997635/128667199-8d6d1cc4-96cd-426a-8fbe-5c6ee99fe1b8.png)

⬇️ dt를 양변에 곱함

![image](https://user-images.githubusercontent.com/65997635/128667223-3b4796f9-690c-4015-b03d-abee44a95c34.png)


### 비선형 변환

#### 선형 변환의 특징
1. 변환 후에도 원점의 위치 동일
2. 변환 후에도 격자들의 형태가 직서의 형태를 유지
3. 격자 간의 간격이 균등

### 비선형 변환
: 아래의 왼쪽 그림과 같이 비선형 변환을 거치면 격자가 선형적이지 않음

### 자코비안 행렬
: 비선형 변환 후 얻은 좌표를 미소 영역으로 본다면 오른쪽 그림과 같이 근사할 수 있음

![image](https://user-images.githubusercontent.com/65997635/128667848-73aa4965-0051-4bbc-aa24-b78ff8b5e3c4.png)

- 선형 변환의 특징 1번은 변환하고자 하는 (x,y)에서의 점을 원점이라 생각하면 해결
- 자코비안 행렬식의 의미 : 원래 좌표에서의 면적대비 변환 후 면적 비율

