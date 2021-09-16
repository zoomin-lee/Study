# 1. 캐시가 필요한 이유는?
## Caching
: 컴퓨터 처리 성능을 높이기 위한 기법으로 CPU가 데이터를 처리할 때 메모리와 데이터를 주고 받는데, CPU에 비해서 메모리는 속도가 느리기 때문에 병목 현상을 완화한다

![image](https://user-images.githubusercontent.com/65997635/129351469-54885940-2c3a-4b1c-8420-55c7961175c7.png)

<br/>

## Cache Memory
: CPU와 메모리 사이에 위치하며 메모리 계층 구조에서 register 다음으로 상위에 위치한다.

![image](https://user-images.githubusercontent.com/65997635/129351437-186e5c04-bfa8-49d9-8969-54c581dde813.png)

<br/>

## 기본 구성

![image](https://user-images.githubusercontent.com/65997635/129351507-fbfe968c-03e5-42f6-9c15-e464e51957bc.png)

: 각각 Cache Set ( S )는 E개의 Cache Line으로 구성되어 있는데, 각각의 line은 B bytes의 Block으로 구성된다.

- Cache Size : S x E x B

### Data Block의 구조
- Valid Bit : 해당 line이 의미 있는 정보가 담고 있는지의 여부를 알려줌
- Tag : line에 저장된 block을 식별해주는 역할을 하며, CPU가 요청한 데이터의 주소의 일부로 사용한다. 
- Cache Block : 저장된 메모리

<br/>

<br/>


# 2. Cache hit ratio 란?
: CPU가 메모리에 접근하기 전에 Cache Memory에서 원하는 데이터의 존재여부를 미리 확인하는데, 이때 필요한 데이터가 있는 경우를 적중(hit) 없는 경우를 실패(miss)라고 한다.
- Cache hit ratio = hit 횟수 / 메모리 참조 횟수
- miss rate = 1 - hit rate
- miss rate을 줄이기 위해선 cache 용량을 늘리면 되지만, hit time과 전력소모가 같이 오를 수 있다

<br/>

## Average Memory Access Time( AMAT )
: Memory에 access할 때 소요되는 평균 시간
- AMAT = Hit time + Miss rate x Miss Penalty
- Cache hit일 경우 : memory access가 불필요하므로 hit time만 계산
- Cache miss일 경우 : memory access가 필요하므로 miss에 대한 time 계산을 더해줌

<br/>
메모리 접근하는데 x 사이클이 걸리고 캐시에 접근하는데 y 사이클이 걸리며 캐시 hit rate 가 h %일 때 effective access time은?
: y * h + ( y + x ) * ( 1 - h )

### Cache Miss 이유
- Compulsory miss : 해당 메모리 주소를 처음 불렀기 때문에 나는 캐시 미스. 
- Conflict miss : 캐시 메모리에 A데이터와 B데이터를 저장해야되는데, A와 B가 같은 캐시 메모리 주소에 할당되어서 나는 캐시 미스로 direct mapped cache에서 많이 발생.
- Capacity miss : 캐시 메모리에 공간이 부족해서 나는 캐시 미스.

Block size가 클수록 주변 데이터를 많이 가져오므로 compulsory miss는 줄어들지만, Block 개수 자체는 적어지므로 confilict miss는 커짐


### Hit time
: Cache에 있는 데이터가 Processor로 전달되는 시간

### Miss Penalty 
: Cache에서 Miss가 발생했을 경우,  Main memory에서 데이터를 가져와 Cache 업데이트를 한 후 CPU까지 가져가는데 걸리는 시간
- Block size가 클수록 Main memory에서 가져올 block의 크기가 커지기 때문에 Miss penalty가 커짐

<br/>

## Locality of Reference
: Cahce Memory의 성공 여부는 참조의 지역성 원리에 달려있다.
- 지역성 : 짧은 시간 동안 제한된 주소 공간의 일부만 참조되는 경향

### Temporal Locality
: CPU가 한 번 참조한 데이터는 다시 참조할 가능성이 높다

### Spatial Locality
: CPU가 참조한 데이터와 인접한 데이터가 참조될 가능성이 높다.

<br/>

<br/>

# 3. Cache Memory의 Mapping Process들의 장단점을 비교하여라.
: Cache의 용량이 주기억장치의 용량보다 적기 때문에 주기억장치의 일부분만 캐시로 적재될 수 있음

<br/>

## Direct Mapping 
: 메인 메모리를 일정한 크기의 블록으로 나누고 각각의 블록을 캐시 위 정해진 위치에 매핑하는 방식으로 Cache line은 1이다.
- 장점 : 가장 쉽고 간단
- 단점 : 비어 있는 라인이 있더라도 동일 라인의 메모리 주소에 대하여 하나의 데이터밖에 저장 할 수 없기 때문에 conflict miss가 자주 발생하게 됨

<br/>

## Full Associative Mapping
: 태그 필드를 확장하여 캐시의 어떤 라인과도 무관하게 매핑 시킬수 있는 매핑 방법
- 장점 : 캐시를 효율적으로 사용하게 하여 캐시의 히트율 증가
- 단점 : 특정 캐시 set안에 있는 모든 블럭을 검사하여 원하는 데이터가 있는지 검사해야 함

<br/>

## Set Associative Mapping
: 위의 두 매핑방식의 장점을 취하고 단점을 최소화한 절충안으로 Cache line 개수만큼 같은 tag의 다른 데이터를 담을 수 있음
- 장점 : 하나의 set에 많은 데이터를 저장하므로 conflict miss 확률이 줄어듦
- 단점 : 탐색해야하는 cache line 수가 많아지므로 hit time과 전력 소모가 늘어남

<br/>

<br/>
             
# 4. Cache Memroy의 Write 정책에는 무엇이 있는가?
: 캐시에 저장되어 있는 데이터에 수정이 발생했을 때 그 수정된 내용을 주기억장치에 갱신하기 위해 시기와 방법을 결정하는 것

## Write-Through 
: 캐시에 쓰기 동작이 이루어질 때마다 캐시 메모리와 주기억장치의 내용을 동시에 갱신
- Write buffer : 메모리에 쓰이기 위해 기다리는 동안 데이터를 저장하는 큐
- 장점 : 구조가 단순하며 메모리와 캐시의 데이터를 동일하게 유지하는 데 별도의 신경을 쓰지 않아도 된다.
- 단점 : 데이터에 대한 쓰기 요청을 할 때마다 항상 메인 메모리에 접근해야 하므로 캐시에 의한 접근 시간의 개선이 없어짐
  - 하지만, 실제 프로그램에서 메모리 참조 시 쓰기에 대한 작업은 통계적으로 10~15%에 불과

<br/>

## Write-Back 
: 캐시에 쓰기 동작이 이루어지는 동안은 캐시의 내용만이 갱신되고, 캐시의 내용이 캐시로부터 제거될 때 주기억장치에 복사함
- 장점 : 일한 블록 내에 여러 번 쓰기를 실행하는 경우 캐시에만 여러 번 쓰기를 하고 메인 메모리에는 한 번만 쓰게 되므로 이 경우에 매우 효율적임
- 단점 : cache 내에 변경된 block인지 표시하는 dirty bit가 필요함.

<br/>

<br/>

# 5. Cache의 Replacement Policy에는 어떤 것들이 있는가?
: Cache에 있는 데이터를 지우는 방법

## Least Frequently used ( LFU )
: 자주 사용되지 않는 데이터를 지우는 방식

## Frist in First out ( FIFO )
: 가장 먼저 들어온 데이터를 지우는 방식

## Random Replacement Policy
: random하게 cache에 있는 데이터를 날리는 방식이지만, LRU와 비슷한 효과를 가짐
- Cache는 어차피 Main memory에 비해 사이즈가 무척 작기 때문에 효과적

<br/>
<br/>

# 6. Virtual Memory와 Virtual Address란 무엇인가?

![image](https://user-images.githubusercontent.com/65997635/129380067-ac67ef43-c1bc-46c6-b2c3-69ca174c74fd.png)

## Virtual Memory 
: Physical Memory의 한계를 극복하기 위해서 하드디스크의 일부를 메모리처럼 사용하는 Logical Memory

## Virtual Address
: 프로세스가 가상 메모리를 참조하기 할 때 사용하는 주소
- CPU에서 프로그램은 Logical address의(= Virtual Address)를 사용하지만, 실제 저장되어 있는 내용을 가져올 때는 Physical address로 접근해야함
- Memory Management Unit( MMU ) : Logical address를 physical address로 mapping해주는 hardware device

<br/>
<br/>

# 7. Paging이란?
: Virtual Address 공간을 일정한 크기의 Page로 나누는 것
- Page Frame : Physical Memory가 이런 Page들을 담기 위해 나눠진 영역

<br/>

## Page Table 
: Physical address( 물리적 메모리의 Page Frame )와 Logical address( 가상 주소 공간의 page ) 사이에 상호 변환이 가능한 테이블로 운영체제(OS)가 제공해주며 Main Memory에 저장됨
- Page Entries(PTEs) : 개별 매핑 데이터로 virtual pages와 physical pages의 mapping 관계
- OS는 프로그램이 엉뚱한 Physical address에 쓰려는 경우를 막고 응용 프로그램간의 메모리 충돌이 일어나지 않도록 해준다.
- 프로세스마다 하나씩 존재하게 되며, 메인 메모리 (RAM)에 상주한다.

<br/>

## TLB(Translation lookaside buffer)
: MMU 안의 작은 cache로 자주 쓰는 Page Table의 주소를 저장해 둠으로써, translation의 속도를 향상시켜 줌

- TLB hit : 가상 주소가 물리 메모리 주소로 변환되어야 할 때, 먼저 TLB를 검색한다. TLB에서 검색이 성공하면, 즉시 물리 메모리 주소를 반환하며, 프로세스는 해당 주소로 접근이 가능하다.
- TLB miss :  TLB에서 검색이 실패하면, 통상적으로 핸들러는 해당 가상 주소에 맵핑된 물리 메모리 주소가 있는지 페이지 테이블을 검색하게 된다. (Page walk)

<br/>

## Page Hit
: 접근하고자 하는 가상 주소에 해당하는 PTE의 Valid 비트가 1이면, 해당 가상 페이지가 물리 페이지에 맵핑되어 있음을 의미

<br/>

## Page Miss
: 접근하려는 가상 주소에 해당하는 PTE의 Valid 비트가 0이고 그곳에 디스크의 특정 위치 정보가 저장되어 있다면, 해당 가상 페이지가 물리 페이지에 맵핑되어 있지 않음을 의미
- Page Fault Exception이 발생하여 Page Fault Handler 호출

### Page Fault 순서
페이지 폴트 핸들러는 현재 메인 메모리에서 특정 물리 페이지를 선택하여 추방하고, 디스크에게 요청된 가상 페이지를 가져오도록 명령한 뒤 문맥 전환을 통해 잠시 다른 프로세스에게 제어를 넘겨준다. 이후 디스크가 가상 페이지를 메인 메모리에 로드하는 작업을 완료하면, 인터럽트를 발생시켜서 문맥 전환을 통해 다시 페이지 폴트 핸들러의 프로세스로 제어를 옮긴다. 그러면 페이지 폴트 핸들러는 추방된 물리 페이지와 새로 들어온 물리 페이지의 정보를 바탕으로 PTE를 갱신하고, 페이지 폴트를 일으켰던 명령어의 위치로 다시 리턴하여 해당 명령어를 재실행한다.

### Cache miss와 Page Fault 횟수 비교
- cache miss > page fault
: block size가 page size보다 훨씬 작기 때문에 새로운 block이 새로운 page보다 훨씬 자주 나올 것이다.

### DRAM 캐시와 Cache 메모리 Miss Penalty 비교
- DRAM 캐시(= 디스크의 캐시) > 캐시 메모리(= 메인 메모리의 캐시)
: DRAM과 SRAM의 속도 차이는 10배 정도인 반면, 디스크와 DRAM의 속도 차이는 거의 10,000배에 이르기 때문이다.

<br/>
<br/>

# 8. Process와 Thread의 차이
## Process
: 프로그램이 실행 중인 상태로 특정 메모리 공간에 프로그램의 코드가 적재되고 CPU가 해당 명령어를 하나씩 수행하고 있는 상태
- 각각의 프로세스는 특정 Context에서 실행
- Context : 커널이 잠들어 있는 프로세스를 다시 실행하는 데 필요한 모든 상태 정보( 범용 레지스터, 프로그램 카운터, 상태 레지스터, 유저 스택, 커널 스택, 기타 커널 자료 구조(EX. 페이지 테이블, 프로세스 테이블, 파일 테이블) 등 )

### Process의 상태 
- 실행(Run)상태 : 프로세스가 프로세서를 차지하여 서비스를 받고 있는 상태 
- 준비(Ready)상태 : 실행될 수 있도록 준비되는 상태 
- 대기(Waiting)상태 : CPU의 사용이 아니라 입출력의 사건을 기다리는 상태

### Independent Logical Control Flow
: 자신이 프로세서를 독차지하고 있는 듯한 착각을 제공

### Private Address Space
: 자신이 메모리를 독차지하고 있는 듯한 착각을 제공

<br/>

## Thread
: 운영 체제에서 프로세서 시간을 할당하는 기본 단위

- 운영체제는 하나의 프로세스에서 여러 개의 쓰레드가 수행될 수 있도록 한다.
- 이러한 쓰레드는 같은 프로세스에 있는 자원과 상태를 공유한다. 
- 즉, 같은 프로세스 내에 있는 쓰레드는 같은 주소 공간에 존재하게 되며 동일한 데이터에 접근할 수 있고 하나의 쓰레드가 수정한 메모리는 같은 메모리를 참조하는 쓰레드에 영향을 미치게 된다.

### Context Switching
: 스케줄링에 따라 프로세스를 준비에서 실행으로, 실행에서 대기로, 대기에서 준비로, 실행에서 준비 등의 이렇게 상태가 변경될 때 필요한 내용을 PCB에 저장 내지는 로드 시키도록 하는 과정

### 같은 프로세스 내부의 thread들끼리 전환되는 것과 다른 프로세스간의 thread까리 전환되는 것이 어떻게 다른가?
: 쓰레드는 프로세스의 자원을 공유하여 사용하고 프로세스가 바뀌지 않는 이상 데이터가 그대로 남아있기 때문에 그대로 가져다가 쓰면되지만 프로세스가 변경되게 되면 cache의 정보, 가상 메모리, TLB등의 정보가 모두 지워지기 때문에 데이터 접근하는데 오래걸린다.

- PCB(Process Control Block) : OS에서 제공하는 일종의 데이터베이스로 프로세스의 작업 상태를 저장할 수 있는 공간


<br/>
<br/>

# 9. Exceptional Control Flow에는 어떤 것들이 있나요?

## Exceptional Control Flow 
: sequential하게 수행되는 control flow를 변화시키는 방법

| level            | 설명                                                                                      |
|------------------|-------------------------------------------------------------------------------------------|
| Hardware         | 하드웨어 수준에서 특정 이벤트의 발생이 감지되면 그 이벤트에 해당하는 예외 핸들러로 제어가 이동 |
| Operating System | 커널이 Context Switch을 통해 한 user Process가 가지고 있던 제어를 다른 Process에 넘겨줌     |
| Application      | System call(Trap)을 통해 OS에 진입하여 특정 서비스를 요청하거나, 한 Process가 다른 Process에게 Signal을 전송하여 제어가 수신자 측의 시그널 핸들러로 넘어가도록 함 |

<br/>

## Exception : Low level ECF( Hardware, Operating System level )
: 프로세서는 특정 Event( 프로세서 상태의 변화 )의 발생을 감지하는 순간, Exception Talbe을 참조하여 해당되는 Exception Handler로의 Indirect Procedure Call을 수행함. Exception Handler가 처리를 마치면 아래의 3가지 동작 중 하나를 수행함

- 이때, Exception Handler는 Kernel Code 에서 수행됨
- hardware interrupt는 현재 state과 무관하게 정해진 코드를 수행하지만, software interrupt는 register 상태에 따라 다른 코드를 수행하도록 구현할 수 있다.

![image](https://user-images.githubusercontent.com/65997635/129431481-ceca5e43-8384-4d7d-8b02-6c0a2629622a.png)

### Asynchronous Exception : Interrupt
: 프로세서 외부의 입출력 장치들로부터 전달받는 신호에 의해 발생하는 예외
- Interrupt Handler : 인터럽트에 해당하는 Exception Handler
- Interrupt Handler는 처리가 끝나면 다음 명령어(I_next)로 리턴

### Synchronous Exception : Trap ( System Call ), Fault, Abort
### Trap ( System Call )
: 특정 명령어를 실행하여 의도적으로 발생시키는 예외
- System Call :  일반적인 함수와 유사한 인터페이스로 커널의 서비스를 요청
- System Call Table은 Exception Talbe과 다름!

### Fault
: 특정 명령어의 실행 결과로 초래된 (회복 가능한) 에러에 의해 발생하는 예외
- Fault Handler : Fault에 해당되는 Exception Handler
- Fault Handler는 에러를 고치는 것에 성공하면 현재 명령어(I_curr, Faulting 명령어)로 리턴하고, 에러를 고칠 수 없으면 커널에 존재하는 abort 루틴으로 리턴하여 해당 프로그램을 종료시킨다.
- ex) Page Fault

### Abort
: 특정 명령어의 실행 결과로 초래된 (회복 불가능한) 에러에 의해 발생하는 예외
- Abort Handler : Abort에 해당되는 Exception Handler
- 다른 핸들러들과 달리 원래 프로그램의 실행 흐름으로 리턴하지 않고 무조건 abort 루틴으로 리턴하여 프로그램을 종료

<br/>

## Context Switch : High level ECF( OS Kernel )
: 현재 프로세스의 문맥을 저장하고, 새로 실행할 프로세스의 문맥을 복원하며, 제어를 해당 프로세스로 넘겨주는 것

- Low level ECF에 해당되는 Exception 매커니즘을 기반으로 구현됨

### Interrupt의 결과로 발생
: 예를 들어, 대부분의 시스템들은 주기적으로(1ms ~ 10ms 간격) Timer Interrupt를 발생시키는 메커니즘을 가지고 있다. Timer Interrupt가 발생하면 커널은 현재 프로세스가 너무 오래 실행되었다고 판단하고 새로운 프로세스로의 Context Switch을 수행

### Kernel이 System Call을 수행할 때 발생
: System Call이 특정 이벤트의 발생을 기다린다면, 이는 스케쥴러를 호출하여 현재 프로세스를 잠들게 하고 다른 프로세스에게 제어를 넘겨주는 Context Switch을 수행
