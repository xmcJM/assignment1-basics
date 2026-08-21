(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than 
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various 
input strings.

UTF-8 无需处理字节序、兼容 ASCII，并且通常更节省空间；相比之下，UTF-16 和 UTF-32 常产生大量空字节，还可能需要处理字节序或 BOM

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string 
into a Unicode string. Why is this function incorrect? Provide an example of an input byte 
string that yields incorrect results.
4
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'é'


(c) Give a two-byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation

为什么优先使用 UTF-8
UTF-8 使用变长编码：ASCII 字符占 1 字节，其他字符占 2～4 字节。例如 "A"：
UTF-8：41
UTF-16：00 41 或 41 00
UTF-32：00 00 00 41 或相反顺序
因此 UTF-8 对英文文本更紧凑、兼容 ASCII，也没有 UTF-16/UTF-32 的字节序问题，比较适合基于字节训练 tokenizer。

为什么不能逐字节解码
UTF-8 的一个字符可能跨多个字节。例如 "é" 编码为：
C3 A9
C3 表示一个双字节字符的开头，A9 是后续字节，只有组合起来才能得到 "é"。错误函数分别解码 C3 和 A9，所以会触发 UnicodeDecodeError。

为什么存在非法字节序列
UTF-8 对每个字节的位模式及组合方式有严格限制。C0 AF 虽然可以用过长形式表示 /，但 / 本应直接编码为 2F；UTF-8 禁止这种不唯一的“过长编码”，因此 C0 AF 是非法序列。

d_model 通常就是 Transformer 的 hidden dimension（隐藏维度），也常写作：
hidden_dim
hidden_size
embedding_dim
model_dim
它表示 Transformer 主干中，每个 token 用多少个数表示。例如：
x.shape == (batch_size, sequence_length, d_model)
如果 d_model = 768，那么每个 token 都对应一个 768 维隐藏向量。
不过需要注意，Transformer 中还有其他“隐藏维度”：
d_model：Transformer 主干的隐藏维度。
d_ff：前馈网络中间层的维度，通常比 d_model 大。
head_dim：每个注意力头的维度，通常满足：
