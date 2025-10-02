# Problem (unicode1): Understanding Unicode (1 point)

(a) What Unicode character does chr(0) return?

* chr(0)返回'\x00'，这是null的unicode编码。

(b) How does this character’s string representation (__repr__()) differ from its printed representation?

* 字符的__repr__()返回的是"'\\\x00'"，加上__repr__()后打印的信息是更接近对象真实内容的编码。

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:  >>> chr(0) >>> print(chr(0)) >>> "this is a test" + chr(0) + "string" >>> print("this is a test" + chr(0) + "string")

结果如下所示，在print中，chr(0)是null字符，它被忽视了，而在直接输出中才会出现。

```python
>>> "this is a test "+chr(0)+" string"
'this is a test \x00 string'
>>> print("this is a test "+chr(0)+" string")
this is a test  string
```

# Problem (unicode2): Unicode Encodings (3 points)

(a)  What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

* 因为UTF-8是最广泛使用的编码，兼容性最好
* UTF-8 是可变长编码， UTF-8 对英文（ASCII）字符只占 1 个字节，而UTF-16是“准可变长”编码，大部分常用字符都是2字节，部分特殊字符才用4字节，而utf-32是定长编码，都是4字节。这使得UTF-8 平均字节数最少，因此序列更短。
* UTF-8的可变长度减少了 tokenization 里出现无效的合并。

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):  
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])  
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 
'hello'
```

* 像这种输入就会错误： `decode_utf8_bytes_to_str_wrong("hello你好".encode("utf-8"))`，因为utf-8是变长编码，一个中文占用3个字节，而这里对这3个字节直接单独解码，所以会错误。

(c) Give a two byte sequence that does not decode to any Unicode character(s).  
Deliverable: An example, with a one-sentence explanation.

* b'\xc0\x80' 就是一个无效的utf-8编码，因为两字节的utf-8的合法序列范围是：0xC2–0xDF + 0x80–0xBF。

# Problem (transformer_accounting): Transformer LM resource accounting (5 points)

(a) Consider GPT-2 XL, which has the following configuration:  vocab_size : 50,257  context_length : 1,024  num_layers : 48  d_model : 1,600  27 num_heads : 25  d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

* 可训练参数数量：
  * Embedding: vocab_size * d_model = 50257 * 1600 = 8,0411,200
  * TransformerBlock layers: num_layers * (multi-head attention + feed-forward + 2 * RMSNorm)
    * multi-head attention(Q, K, V, output_linear): 4 * d_model * d_model 
    * multi-head attention(rope): 0
    * feed-forward: 3 * d_model * d_ff
    * RMSNorm: d_model
    * 所以每层的参数量是：4 * 1600 * 1600 + 3 * 1600 * 6400 + 2 * 1600 = 10,240,000 + 30,720,000 + 3,200 = 40,963,200
    * 总的TransformerBlock参数量是：48 * 40963200 = 1,966,233,600
  * ln_final RMSNorm: d_model = 1600
  * lm_head Linear:  d_model * vocab_size = 1,600 * 50,257 = 80,411,200
  * 所以总的可训练参数量是：80,411,200 + 1,966,233,600 + 1,600 + 80,411,200 = 2,127,065,600 ≈ 2.13B
* 对于single-precision floating point，每个参数占4字节，所以总内存需求是：2,127,065,600 * 4 = 8,508,262,400 字节 ≈ 8.51 GB。

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

* Embedding: 0, 主要是内存拷贝的开销
* TransformerBlock layers: num_layers * (multi-head attention + feed-forward + 2 * RMSNorm)
  * multi-head attention projections (Q,K,V,W_o): 4 * 2 * context_length * d_model * d_model = 4 * 2 * 1024 * 1600 * 1600 = 20,971,520,000
  * multi-head attention quadratic matmuls (QKV): 3 * 2 * context_length * context_length * d_model = 3 * 2 * 1024 * 1024 * 1600 =10,066,329,600
  * multi-head attention(output_linear): 2 * context_length * d_model * d_model = 2 * 1024 * 1600 * 1600 = 5,242,880,000
  * multi-head attention(rope for Q, K): 2 * 3 * context_length * d_model = 2 * 3 * 1024 * 1600 = 9,830,400
  * feed-forward: 3 * 2 * context_length * d_model * d_ff = 3 * 2 * 1024 * 1600 * 6400 = 62,914,560,000
  * RMSNorm: ≈ 4 * context_length * d_model = 4 * 1024 * 1600 = 6,553,600
  * 所以每层的FLOPs是：20,971,520,000 + 10,066,329,600 + 5,242,880,000 + 9,830,400 + 62,914,560,000 + 6,553,600 = 99,211,673,600
  * 总的TransformerBlock FLOPs是：48 * 99,211,673,600 = 4,762,160,332,800
* ln_final RMSNorm: 4 * context_length * d_model = 4 * 1024 * 1600 = 6,553,600
* lm_head Linear: 2 * context_length * d_model * vocab_size = 2 * 1024 * 1600 * 50257 = 164,682,137,600
* 所以总的FLOPs是： 0 + 4,762,160,332,800 + 6,553,600 + 164,682,137,600 = 4,926,849,024,000 ≈ 4.93 TFLOPs

(c) Based on your analysis above, which parts of the model require the most FLOPs?

* TransformerBlock layers中的feed-forward部分需要最多的FLOPs。

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

* GPT-2 small:
  * 可训练参数数量：190,460,160 ≈ 190M
  * 总的FLOPs：383,550,750,720 = 0.384 TFLOPs
  * 383550750720 = 0.384 TFLOPs
  * multi-head attention projections占总FLOPs的比例：15.1%
  * multi-head attention quadratic matmuls占总FLOPs的比例: 15.1%
  * feed-forward占总FLOPs的比例: 45.4%
* GPT-2 medium:
  * 可训练参数数量：505,629,696 ≈ 505.63 M
  * 总的FLOPs：1,136,444,571,648 ≈ 1.136 TFLOPs
  * multi-head attention projections占总FLOPs的比例：18.1%
  * multi-head attention quadratic matmuls占总FLOPs的比例: 13.6%
  * feed-forward占总FLOPs的比例: 54.5%
* GPT-2 large:
  * 可训练参数数量：1,072,469,760 ≈ 1.07247 B
  * 总的FLOPs：2,475,664,343,040 ≈ 2.476 TFLOPs
  * multi-head attention projections占总FLOPs的比例：19.5%
  * multi-head attention quadratic matmuls占总FLOPs的比例: 11.7%
  * feed-forward占总FLOPs的比例: 58.6%
* 随着模型规模的增加，TransformerBlock layers中的feed-forward部分占总FLOPs的比例显著增加。

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

* 如何context length变为原本的16倍，总FLOPs会变为1.948 × 10¹⁴ (≈ 194.79 TFLOPs)，是原本的39.56倍。其中multi-head attention projections增长为原本的16倍，feed-forward增长为16倍，而multi-head attention quadratic matmuls增长为256倍。
  * Attention projections从原本的 1 T FLOPs (总Flops的20.3%) → 16 T FLOPs (总Flops的8.2%)
  * Attention quadratic从原本的 483.18 G FLOPs (总Flops的9.8%) → 123.70 T FLOPs (总Flops的63.5%)
  * Feed-forward从原本的 3.01 T FLOPs (总Flops的61.0%) → 48.32 T FLOPs (总Flops的24.8%)
