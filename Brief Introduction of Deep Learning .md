# Three steps of Deep Learning
- Define a set of function
- Goodness of function
- Pick the best function

# Neural
- 一個 Neural 做的事情就是根據 input，乘上 weight 相加後，加上一個 bias，再通過 activation function 計算出 output

![image](https://raw.githubusercontent.com/kevingo/ml-az/master/screenshots/neural.png)

# Neural Network
- 不同的連接方法會產生不同的 network structure
- 每一個 neural network 都有自己的 weight 和 bias，這些通通集合起來叫做 parameters
 
 ![image](https://github.com/kevingo/ml-az/raw/cf894b563c3ae96df155adaa0b0124f68f6377fc/screenshots/neuralnetwork1.png)

# Fully connected Feedforward Network
- 當一個 network 的 weight 和 bias 都知道了之後，他就是一個 function，input 是一個 vector，output 也是一個 vector
- 如果我們不知道參數，只知道 network 連接的方式，這樣的 network 就是定義了一個 function set
- 因為每個 layer 之間倆倆之間都有連接，所以叫「Fully Connceted」。因為值的傳遞方式是由 layer 1 到 layer 2 ... ，所以叫「Feedforward」。

![image]()

# Deep learning
- Deep = Many hidden layer

# Network Operation
- Network 的運作常用 matrix operation 來表示

![matrix](https://github.com/kevingo/ml-az/blob/master/screenshots/matrixoperation.png?raw=true)

- 計算的方式就是把 input * weight + bias，再通過 activation function 後，得到 output
- 轉換成 matrix operation 的原因是因為這樣可以透過 GPU 來加速運算
	- Ref: [Understanding the Efficiency of GPU Algorithms for
Matrix-Matrix Multiplication](https://graphics.stanford.edu/papers/gpumatrixmult/gpumatrixmult.pdf) 

![matrix](https://github.com/kevingo/ml-az/blob/master/screenshots/matrixoperation2.png?raw=true)

# Output layer
- 我們可以把在 output layer 之前的過程當作是 feature extractor，這取代了之前需要人工抽 feature 做 feature engineering 的過程
- 所以 x1, x2 ... xk 可以想像成一組新的 feature
- Output layer 就可以想像成一組 multi-class 的 classfier，這個 classifier 用的 feature 不是直接拿 input ，而是拿經過多個 hidden layer 所抽取出來，可被 seperable 的 x1 ... xk
- multi-class classfier 要通過一個 softmax function 來產生 output

![outputlayer](https://github.com/kevingo/ml-az/blob/master/screenshots/outputlayer.png?raw=true)

# Example Application
- 比如說我們用手寫辨識當範例
- Input 可以是一個 256 維的 vector，每一個維度是 0（代表該格沒有塗黑) 或是 1 （代表該隔有塗黑)
- Output 可以是 y1 ~ y10 的 10 維 vector，分別代表便是為 1 ~ 10 的機率值
- 所以我們就可以把手寫辨識 format 成一個 input 是 256 維，output 是 10 個維度的 feedforward network 的 problem

![app](https://github.com/kevingo/ml-az/blob/master/screenshots/exampleapp1.png?raw=true)

- 事實上，因為我們的 input 和 output 已經定義好了，所以要如何建構中間的 hidden layer 變成關鍵因素。對於 neural network 來說，我們現在唯一的 constraint 只有 input 要有 256 維度、output 要有 10 維度，但是中間要有幾個 hidden layer，每一個 hidden layer 要有多少 neural，是沒有限制的，你必須要自己去設計他。如果我今天決定了一個很差的 function set，那就勢必找不到一個好的結果了。

![app](https://github.com/kevingo/ml-az/blob/master/screenshots/exampleapp2.png?raw=true)

- 事實上，要怎麼決定 network structure 是很難的，通常要憑直覺或多方的嘗試。
- 過去我們常常需要做 feature engineering 去找一組好的 feature 給 machine 去學習，但是現在我們改用 deep learning 的方式，往往不需要找到好的 feature。比如說，我們在做影像辨識的時候，可以直接把 pixel 丟進去給機器去學。
- 過去在做影像辨識的時候，需要對影像去抽一些人訂的 feature，這件事情(feature transform)
- 但是 deep learning 產生新的問題，就是你要去決定 network structure。
- 所以問題從「怎麼抽 feature」轉為「怎麼定義 network structure」
- Deep learning 是不是好用取決於你覺得抽 feature 比較容易還是定義 network structure 比較容易
- 如果是語音辨識或影像辨識，design network structure 可能比定義 feature 容易
- 因為這類問題人類太容易做這些事情了，容易到有點像是直覺，所以很難直覺地去抽取適當的 feature 出來給 machine 學習

# Goodness of function
- 怎麼決定一組參數的好壞呢？
- 假設給訂一組參數，給定一組 input，我們可以得到 output y1 ~ y10。接著對每一個 input，我們會有一個標準答案(target)。針對這個標準答案，我們可以算出一個 cross entropy 

![image](https://github.com/kevingo/ml-az/blob/master/screenshots/lossofexample.png?raw=true)

- 我們不會只有一組 input，所以對於所有的 input 來說，我們的目標就是找到一個 function 或參數集合，minimum 我們的 total loss

![image](https://github.com/kevingo/ml-az/blob/master/screenshots/totalloss.png?raw=true)

# Gradient Descent（梯度下降算法)
- Gradient 會告訴我們往哪個方向會增加計算出來的 output，我們的目標是要最小化 total loss function，所以要往反方向走（加個負號)
- 要 minimum total loss 所用的方法就是 gradient descent
- 做法是，首先先隨機給訂一組初始參數，接著去計算他的 gradient(去計算每個參數對 total loss 的偏微分)，把這些偏微分集合起來叫做 gradient
- 接著可以更新參數，把每個參數去減掉一個 learning rate 得到新的參數
- 接著就反覆以上步驟就可以找到一組比較好的參數

# Backpropagation
- Back Propagation 利用 Chain Rule 的特性，反向的更新權重
- 現在有許多的 tools 可以幫助你算 backpropagation(有效計算偏微分)，因為我們有許多的參數，如果要替每一個參數去算偏微分是相當花時間的

![image](https://github.com/kevingo/ml-az/blob/master/screenshots/backpropagation.png?raw=true)


