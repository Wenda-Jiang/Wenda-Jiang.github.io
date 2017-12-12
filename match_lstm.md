# Match-LSTM - 利用Attention机制，带着问题阅读原文
[MACHINE COMPREHENSION USING MATCH-LSTM AND ANSWER POINTER](https://arxiv.org/pdf/1608.07905.pdf)
## 摘要
本文介绍一种结合 math-LSTM 和Pointer Net利用end-end的来解决QA问题的方式

## 模型
最主要的还是 match-LSTM：有两个句子，一个是前提，另外一个是假设，match-LSTM序列化的经过假设的每一个词，然后预测前提是否继承自假设。

简单的说：带着问题去阅读原文，然后用得到的信息去回答问题

1. 先利用LSTM阅读一遍passage，得到输出的encoding 序列
2. 然后带着question的信息，重新将passage的每个词输入LSTM，再次得到passage的encoding信息。但是这次的输入不仅仅只有passage的信息，还包含这个词和question的关联信息，它和qustion的关联信息的计算方式就是我们在seq2seq模型里面最常用的attention机制。
3. 然后将信息输入answer模块，生成答案

下面介绍详细的模型
### 1. 预处理层LSTM Preprocessing层

首先对文本和问题分别单独用LSTM进行单向的encoder
$$
\begin{matrix}
H^p=LSTM(P)  \newline
H^q=LSTM(Q)  \newline
\end{matrix}
$$
$H^p \in R^{[l, p]}, H^q \in R^{[l, q]}$
l 是LSTMcell的隐藏层大小，p和q分别是文本passage 和 问题question的长度
代码很简单，两个序列分别经过LSTM序列模型，就得到encoder向量。

```python
lstm_cell_question = tf.nn.rnn_cell_impl.BasicLSTMCell(l, state_is_tuple=True)
encoded_question, q_rep = tf.nn.dynamic_rnn(lstm_cell_question, question,masks_question,dtype=tf.float32)

lstm_cell_passage = tf.nn.rnn_cell_impl.BasicLSTMCell(l, state_is_tuple=True)
encoded_passage, p_rep = tf.nn.dynamic_rnn(lstm_cell_passage, passage,masks_passage, dtype=tf.float32)
```

### 2. Match-LSTM 层

带着qustion来阅读passage，利用的是利用了 [Bahdanau Attention机制](https://arxiv.org/pdf/1409.0473.pdf)机制，具体可以见该论文。

但是为了详细描述，在这里还是详细的描述一遍：

整体的思路可以看作我们在decoder passage，我们聚焦的是qustion向量：
$h^r_i = LSTM(z_i,h^r_{i-1})$

由attention机制我们可以知道，这里的$z_i$是融合passage的input和对qustion的attention信息：
$$ z_i =  \left[
\begin{matrix}
 h_i^p      \newline
 f(H^q)     \newline
\end{matrix}
\right]
$$

$H^p$是prcess层利用LSTM将passage预处理后得到的，
第i个词的向量为$h^p_i \in R^l$，我们在$h^p_i$之后加一个qustion相关的信息，

令：$f(H^q)=H^q\alpha_i$

其中$\alpha_i$是文本passage里面的第i个词，首先计算第i个词和question里面每一个词的相关性权重

$\alpha_i$就是attention的alignment model：

$$
\begin{matrix}
G_i=tanh(W^qH^q+(W^p h^p_i + W^r h^r_{i-1} + b^p) \bigotimes e_Q) \newline
\alpha_i = softmax(w^tG_i + b\bigotimes e_Q) 
\end{matrix}
$$

```python
# tensorflow 里面有现成的BahdanauAttention类
match_lstm = BahdanauAttention(l, q)
```

这样我们就得到了$\alpha_i$

这样我们可以完整的迭代这个序列模型：
$$h^r_i = LSTM(z_i,h^r_{i-1})$$

同理我们将passage倒叙，可以得到倒叙的LSTM模型

$$ \hat{h_i^r} = LSTM(\hat{z_i}, \hat{h^r_{i-1}}) $$

我们令forward和backward得到的转台矩阵分别为$H^r,\hat{H^r}$, 我们把两个矩阵直接连接起来得到最终的状态矩阵
$$ H_r= \left[
\begin{matrix}
 H^r      \newline
 \hat{H^r}     \newline
\end{matrix}
\right]
$$
$H_r \in R^{[2l, p]}$

```python
# LSTM Cell
cell = BasicLSTMCell(l, state_is_tuple=True)
lstm_attender = AttentionWrapper(cell, match_lstm)
reverse_encoded_passage = _reverse(encoded_passage)

# bi-dir LSTM
output_attender_fw, _ = tf.nn.dynamic_rnn(lstm_attender, encoded_passage, dtype=tf.float32, scope="rnn")
output_attender_bw, _ = tf.nn.dynamic_rnn(lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope="rnn")

output_attender_bw = _reverse(output_attender_bw)

# concat
output_attender = tf.concat([output_attender_fw, output_attender_bw], axis=-1)
```


### 3. Answer Pointer

Answer Pointer的思想是从Pointer Net得到的，
它将$H^r$作为输入，生成答案有两种方式：
1. sequence，自动生成答案序列, 序列里面的词是从passage里面选取出来的
2. boundary，答案从passage里面截取，模型生成的是开始和结束下标

#### Sequence
假设我们的答案序列为：
$a=(a_1,a_2,...)$
其中$a_i$为选择出来答案的词在原文passage里面的下标位置，
$a_i \in [1, P + 1]$, 其中第P + 1 是一个特殊的字符，表示答案的终止，当预测出来的词是终止字符时，结束答案生成。

简单的方式是像机器翻译一样，直接利用LSTM做decoder处理：
假设$a_1,a_2,..,a_{k-1}$
$$
\begin{matrix}
O_k = LSTM(a_{k_1}, h_{k-1}) \newline
p(a_k|O_k)=argmax_{P + 1} (softmax(WO_k))
\end{matrix}
$$
找到passage里面概率最大的词的就可以了

这里也利用上节讲的Bahdanau Attention机制，
在预测第k个答案的词时，我们先计算出一个权重向量
$\beta_k$用来表示在[1, P+1]位置的词，各个词的权重

先得到隐藏向量：
$h_k^a=LSTM([H^r;0], h_{k-1}^a )$

计算权重：
$$
\begin{matrix}
F_k=tanh(V[H^r;0] + (W^ah_{k-1}^a + b^a)\bigotimes e_{P + 1} ) \newline
\beta_k = softmax(v^tF_k + c \bigotimes e_{P + 1}) 
\end{matrix}
$$

然后$p(a_k=j)=\beta_{k,j}$

$\beta_{k,j}$最大的一个下标就是$a_k$的值

代码和match-LSTM基本一致

#### Boundary

这种模型很简单，答案$a=(a_s,a_e)$只有两个值

$P(a|H_r)=P(a_s|H_r)p(a_e|a_s,H_r )$
实际过程中，可以直接用
$P(a|H_r)=P(a_s|H_r)P(a_e|H_r)$，选取概率最大的$a_s, a_e$pair
计算方式和上面sequene模型一样，只是有两点不同

1. 没有显示的结束标记，所有在LSTM input的时候不要补0 
2. $a_s \le a_e$


## 实验
使用预训练过的词向量
Boundary模式的answer层忧郁是固定的，可以用bi-dir LSTM
Boundary模式要注意避免截取的段落过长



## 后续改进版本
[Question Answering Using Match-LSTM and Answer Pointer](https://pdfs.semanticscholar.org/f9f0/8511f77c29ff948e146434dfb23608d3deb5.pdf)


### 1. 预处理层LSTM Preprocessing层
单向LSTM -> BiLSTM

$$
H_q=
\left[
\begin{matrix}
h_q^{for} \newline
h_q^{back}
\end{matrix}
\right]
$$

$$
H_p=
\left[
\begin{matrix}
h_p^{for} \newline
h_p^{back}
\end{matrix}
\right]
$$

### 2. regularization
在所有LSTM外面都加上Dropout，droupout概率为0.2

### 3. 调整op，size大小
不同的optimal Adam -> AdaMax
不同的batch-szie 
gradient clipping =0.5

### 实验效果
![](media/15123803633991/15126399426010.jpg)￼

图中可以看出，虽然改进的Final Match-LSTM 在训练集上面指标下降，但是在测试集上面是有提升的
其实在训练过程中dropout和batch_normal真的是很好的两种方式


[Extending Match-LSTM](https://pdfs.semanticscholar.org/3f0f/ce70ab5122ce742daabb8f81473fcb633c85.pdf)

这个工作并没有对Match-Lstm层做任何改动

### 1. 预处理层LSTM Preprocessing层增加attention

P，Q互为attention聚焦
$$
\begin{matrix}
H^p=LSTM(P, attention(Q))  \newline
H^q=LSTM(Q, attention(P))  \newline
\end{matrix}
$$

这个改进其实更近一步，在预处理阶段就带着对方的信息处理每一个词

### 2. Conditional Span Decoder

在上文的处理中，answer start 和 end是独立处理的，这里还是恢复以来关系：

$P(a_e=i,a_s=j|H_r)=P(a_e=i|H_r)=P(a_s=j|H_r)P(a_e=i|a_s=j,H_r)$

其实如果做到以来，直接用LSTM其实就是上述的概率
这里作者做了些改进：
$H_r$经过softmax后得到$a_s$的概率向量$h_{s,j}$
$p(a_s=j|H_r)=h_{s,j}=softMax[H_r])$ 

然后$p_{s,j}$和$H_r$一起作为LSTM的输入

$h_{e,j}=LSTM([p_{s,j},H_r] ))$

$p(a_e=i,a_s=j|H_r)=softmax(h_{e,j})$

![](media/15123803633991/15126452871063.jpg)￼

graph 对比图




