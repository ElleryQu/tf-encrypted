# tf-encrypted

[toc]

## 快速实验

### Mnist

**场景**：

* **训练阶段**：model-owner 在本地训练模型（使用tensorflow），然后将模型的权重秘密分享给三个计算服务器 server{i}。
* **预测阶段**：prediction-client 将数据秘密共享给 server{i}，server{i} 本地计算，将结果发回给 prediction-client，prediction-client open掉结果。

用户配置 config.json：

```json
{
    "server0": "127.0.0.1:20221",
    "server1": "127.0.0.1:20222",
    "server2": "127.0.0.1:20223",
    "model-owner": "127.0.0.1:20224",
    "prediction-client": "127.0.0.1:20225"
}
```

打开五个shell，分别执行命令：

```powershell
conda activate tfe

python -m  tf_encrypted.player server0 --config config.json
python -m  tf_encrypted.player server1 --config config.json
python -m  tf_encrypted.player server2 --config config.json
python -m  tf_encrypted.player model-owner --config config.json 
python -m  tf_encrypted.player prediction-client --config config.json
```

**总结**：

1. 明文操作使用tf定义。
2. 密文操作使用tfe定义，由装饰器@tfe.local_computation包装或直接调用tfe.keras相关接口。
3. 训练预测过程全部写在一个python脚本里，由 master 参与方执行（config.json 定义的第一个参与方）。所有参与方必须能够与 master 通讯。
3. 脚本流程：1. 读入config.json或者本地执行，生成config对象，设置config，将协议置空；2. 实例化训练者和预测者，

### SecureNN

用户配置 config.json：

```json
{
    "server0": "127.0.0.1:20221",
    "server1": "127.0.0.1:20222",
    "crypto-producer": "127.0.0.1:20223",
    "model-trainer": "127.0.0.1:20224",
    "prediction-client": "127.0.0.1:20225"
}
```

打开五个shell，分别执行命令：

```powershell
conda activate tfe

python -m  tf_encrypted.player server0 --config config.json
python -m  tf_encrypted.player server1 --config config.json
python -m  tf_encrypted.player crypto-producer --config config.json
python -m  tf_encrypted.player model-trainer --config config.json 
python -m  tf_encrypted.player prediction-client --config config.json
```

在本地环境下能跑通，但是配置config.json就不行了，奇了怪了。

* 某个全局变量的assign错误：Assign requires shapes of both tensors to match. lhs shape= [784,512] rhs shape= [784,128]。
* cache错误。
  * cache(w0): 		784, 128 → 784, 512		
  * cache_1(b0): 	128, → 512			 	      
  * cache_2(w1):	 128, 128 → 512, 10   happen with cache_3
  * cache_3(b1):	 128, → 10                    happen with cache_2
  * cache_4(w2):	 128, 10 →
  * cache_5(b2):	 10, →
* 解决方案：关掉python进程就可以。这个player的分配有点问题，或者说我没有理解tf.device()；似乎device一旦被分配到某个端口，运行一个实验后该端口会被捆绑模型。优先级不高，以后再研究。

## 项目结构

```
tf-encrypted
|---convert				The TFE Converter.
|---keras				TFE高层封装，采用keras api。 
|---layers				TFE神经网络层的封装。
|---operations			TFE客户端API，用于自定义操作？。
|---player				TFE参与方（Player）的抽象类。
|---protocol			TFE协议，目前包含aby3、SecureNN、   
|	|					Pond的实现。(核心)
|---queue				队列数据结构。通讯与分布式计算相关。
|---serving				通讯与分布式计算相关。
|---tensor				特定数据类型的Tensor。(核心)
|	|					借助原生tf数据类型或自定义dtype实现。
|	|---factory.py		抽象tfe张量、常量、占位符、变量、工厂。
|	|---boolfactory.py	tfe布尔类型张量。基于tf.bool。
|	|---int100.py		tfe基于CRT的浮点数张量。
|	|					基于某种整数类型INT_TYPE。
|---__init__.py
|---config.py
|---private_model.py
|---session.py
|---utils.py
```

## 张量

np.array() → tf.tensor() → tfe.tensor()，三层包装。

为什么要使用CRT增强浮点数表示？

## 主目录

### protocol

#### Pond

一揽子设置。

```python
TFEData = Union[np.ndarray, tf.Tensor]
TFEVariable = Union["PondPublicVariable", "PondPrivateVariable", tf.Variable]
TFEPublicTensor = NewType("TFEPublicTensor", "PondPublicTensor")
TFETensor = Union[TFEPublicTensor, "PondPrivateTensor", "PondMaskedTensor"]
TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_INT_TYPES = [tf.int8, tf.int16, tf.int32, tf.int64]
TripleSourceOrPlayer = Union[TripleSource, Player]
```

一些类。

![image-20220715111641564](D:\WorkStation\tf-encrypted\.a_eq_note\tfe.assets\image-20220715111641564.png)

**class Pond(Protocol)**

初始化：设置server0、server1，设置三元组产生方，设置工厂，设置浮点数类型，

### tensor

#### factory.py

#### native.py

tfe数据类型，向下对接tf的tensor、Variable、Constant，向上为tfe提供基础数据类型支持。

### __init__.py

主要进行以下步骤：

1. 包检查；
2. 将所有协议的公共方法都登记下来；
3. 将当前协议设置为Pond；
4. 声明此模块的可用子模块。
