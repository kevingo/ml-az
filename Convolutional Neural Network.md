# Why CNN for Image
- CNN 經常用在影像處理上
- 我們在 train neural network 的時候，會期待每一個 neural 是一個最基本的 classifier
- 比如說，第一層的 neural 是最簡單的 classifier，他做的事情是找出有沒有綠色出現，有沒有黃色出現，有沒有斜的條紋。第二層的 classifier 是根據第一層的 output，找出有沒有窗框出現、有沒有木紋，再根據第二層的 output，第三層的 neural 可以做到：看到人的上半身就被 activate，看到車子的輪胎就被 activate。


# 為什麼一般的 Fully-Connected Neural Network 不容易做到上面這件事情？
- 用一般的 fully-connected network 往往需要用到太多的參數
- 所以 CNN 其實是利用人的 prior knowledge 把一些參數拿掉，整個模型是比一般的 DNN 來得簡單

# CNN 做影像辨識的觀察
- 每個 neuron 並不需要看完整張圖才能辨識，例如：我們並不需要看完整張圖才知道這是一個鳥嘴
- 同樣的 pattern，可能會出現在 image 不同的部分，但是他是代表一樣的含義，例如：下面這兩張圖，在左上角的鳥嘴和中央的鳥嘴對於 CNN 來說，不需要兩個 neuron 來做辨識的任務，所以我們可以要求這兩個 neuron 用同一組參數

![cnnforimage](https://raw.githubusercontent.com/kevingo/ml-az/master/screenshots/cnnforimage.png)

