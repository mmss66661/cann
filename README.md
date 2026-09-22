# AddRmsNormBias 初赛实现

提交文件为 [`project/kernel.asc`](project/kernel.asc)。它实现残差加、最后一维 RMSNorm 与偏置加法融合，支持 float32、float16、bfloat16。

## 对接方式

代码已对齐工作区下载包中的 `main.asc` 签名及 dtype 编码。`TensorInfo` 与 `TensorGroupInfo` 由 `main.asc` 定义，提交时仅上传可编辑的 `kernel.asc`。

在 `project/` 目录保留平台的 `main.asc`、`CMakeLists.txt`、`run.sh` 和 `scripts/`，于 CANN 9.0.0、Atlas A2/910B 环境执行 `bash run.sh`。这台 Windows 工作机没有 CANN 或 NPU，不能在此确认编译、15 点精度或设备性能。

## 当前提交文件（v9：整批归一化布局，待在线评测）

2026-09-22：以 Pass 54.80 分的 v5 为基线，优化短行批处理的指令调用次数。v8 的流水线实验已退出当前提交文件，源码另存。

- 为 BatchKernel 增加 DENSE 模板路径。gamma/bias 每核展开一次并缓存为整批布局；每批归约后，把 RMS 系数压紧为“每 64 个数据对应 8 个相同系数”的数组，复用已失效的平方临时缓冲。单次跨批 Div 后使用整批 Mul 和 Add。
- D=1024 时，归一化及仿射阶段从 48 次向量 API 调用变为 5 次（2 次系数展开、Div/Mul/Add 各一次）。此计数不包含残差加、归约、搬运、缓存初始化或编译器内部生成的指令，不代表耗时比。
- 新路径适用于 128<D<=1024 且每核至少两批的场景。FP32 每批最多 4096 元素并直接写输出队列；FP16/BF16 每批最多 6144 元素，为完整参数缓存留出 UB。其他场景使用 v5 路径。
- 保持原有运算顺序：FP32 残差加、均方/Sqrt、除 RMS、乘 gamma、加 bias、最后转输出类型。没有引入 Rsqrt 近似、融合乘加或低精度中间结果。
- 共 14 项 CPU 模型检查通过。新增检查覆盖所有 129..1024 宽度的系数/参数地址映射、部分尾批、跨批参数缓存稳定性、三种 dtype 与 golden 的对比、NaN/Inf 的行内传播和 UB 边界。新路径显式 UB 峰值 173184 字节，全路径峰值仍为 184608 字节。不能替代 CANN 编译和 NPU 评测。
- 提交文件 `project/kernel.asc`，快照 `versions/kernel_v9_dense_candidate.asc`，回退版本 `versions/kernel_pass_54_80.asc`。

## 已评测实验（v8：双缓冲流水线，Pass 54.58 分）

2026-09-22：以已经在线 Pass、54.80 分的 v5 为基线，仅增加中等宽度路径的流水线版本。开始本轮时的 v7 源码已原样保存在 `versions/kernel_before_v8_20260922.asc`。

- WideBatchKernel 的新模板分支使用两份输入缓冲、两份输出缓冲，计算当前批之前预取下一批，让不同流水具备重叠执行的条件。队列管理同步与缓冲复用。
- Host 根据 dtype、补齐行宽和 176 KiB 显式 UB 预算选择每批行数；只有可容纳完整行且每核任务超过一批时才启用。输入队列深度为 2，输出队列深度为 1，二者 buffer 数量均为 2。
- 维持 v5 的 FP32 运算、归约顺序、参数缓存、非对齐写回及核间分组；小尺寸和很长的行继续走 v5 原路径。本轮不引入 v6/v7 的重读长行或 dtype 切换实验。
- 10 项 CPU 检查通过，包括新批大小下的双缓冲索引、尾批、写回范围、golden 对比和逐宽度内存检查。新流水线分支最高显式 UB 分配 179776 字节，全部路径最高仍为 184608 字节。CPU 模型不能验证 Ascend 编译、硬件同步或耗时。
- 用户反馈全部 Pass、54.58 分（2026-09-22 14:35:23）；第 8/9 点为 42.17/72.10 us，第 4/5 点为 11.32/17.10 us。未超过 v5，当前已回到 v5 上开展 v9 实验。已通过源码保存在 `versions/kernel_pass_54_58.asc`。
- 设计参考：[Ascend C DoubleBuffer](https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/vector_programming/double_buffer_scenario.html) 和 [TQue 深度与 buffer 数量](https://asc.gitcode.com/api/SIMD-API/basic_api/resource_management/TQue/TQue_intro.html)。

## 历史实验（v7 修订，先前记录，未在本次核实评测）

首版 v7（Div→Mul + RELOAD + FP16/BF16 中宽度 RetainedKernel）在线评测 54.05 分，比 v5 的 54.80 回退。逐点对比后定位到两处负优化并修订：

- **回退小 D 路径的 Div→Mul**（TinyKernel/BatchKernel）：极小 D（case 1-3）时逐元素 `Div` 与 `Mul` 差距可忽略，而「标量倒数」额外引入 Duplicate + 标量 Div 两条指令，反而回退；BatchKernel 的 repeat 形式 Div 已足够高效。已恢复 v5 原实现。
- **保留长行 RELOAD**（case 11/13 改善 -3.04/-0.68μs）：RetainedKernel 增加 `RELOAD` 模板参数，长行（D>6144/8192）且每核少量行时用 6144/8192 大 chunk 两遍读。
- **FP16/BF16 中宽度 RetainedKernel 加 rows 门槛**：D=1025..8192 且 `rows <= cores*2`（平均每核 ≤2 行）时用整行保留的 RetainedKernel（case 8/10 改善 -1.13/-2.29μs），行多时仍走 WideBatchKernel（消除 case 12 的 +1.38μs 回退）。
- 完整 8 项 unittest 通过；显式 UB 分配峰值 184608 字节。
- 修订版尚未在线评测；若编译或精度报错，回退 `versions/kernel_pass_54_80.asc`。

## 历史实验（v6，Pass 54.09 分，已回退）

- 保留 v5 的 FP32 WideBatch 路径，避免回退第 9/12 点；FP16/BF16 在 D=1025..8192 时改用保留整行的直接除 RMS 路径，针对 v3 中第 8/10 点的回退做定向恢复。
- 对 D 超过中等宽度且每核只有少量长行的场景，使用 6144（FP32）或 8192（FP16/BF16）元素的两遍分块路径，降低单个 2048 分块的搬运和归约次数；其余长行继续使用 v5 的路径。
- 新增 6145、8193、12287/12288/12289、16384、24576、32767/32768 等边界的 CPU 精度检查；共 8 项检查通过，显式 UB 分配峰值仍为 184608 字节。
- 用户反馈 v6 全部 Pass、54.09 分，第 12 点由 v5 的 93.91 us 增加到 115.16 us；随后已经按用户要求回退 v5。

## 已通过版本（v5，54.80 分）

- dtype 与算法路径在 Host 侧选定，使用模板 kernel 分别编译各 dtype/路径，减小单个设备入口的指令体积。
- 每核一行的小输入对 D=64/128/256/512/1024 做编译期宽度特化；64 元素归约直接使用 WholeReduceSum。
- WideBatchKernel 的补齐宽度是 512 倍数且不大于 4096 时，两次 WholeReduceSum 完成整批归约，替代批内逐行归约。其余宽度保留原算法。
- 7 项 CPU 检查通过；补充 1536/2560/3584 等非 2 的幂归约宽度以及邻近非对齐尺寸。显式 UB 分配峰值不变。用户反馈 v5 全部 Pass，54.80 分。

## 已通过版本（v4，54.39 分）

- 为中等宽度加入 WideBatchKernel：FP32 的 1024<D<=6144、FP16/BF16 的 1024<D<=8192，一批容纳最多 8192 个补齐后的元素，同时复用每核 gamma/bias 缓存。
- FP32 在输出队列中直接计算，省掉独立 y 缓冲及末尾复制。多行共享均值、开根号、倒数的向量指令；每行的归一化系数仍独立。
- 短行 pitch 为 512/1024 时，使用两级 WholeReduceSum 替换逐组累加。
- 中、长行先计算每行的 FP32 RMS 倒数，然后用 Mul 归一化，减少元素级 Div。未引入 Rsqrt 或低精度内部计算。
- 用户反馈 v4 全部 Pass，54.39 分；第 9/12 点降至 73.42/96.53 us，第 8/10 点回退至 42.83/70.74 us。峰值显式 UB 分配 184608 字节。

## 已通过版本（v3，52.37 分）

- 快速路径现覆盖所有 D<=32768，包括非 64 倍数。GM 中的数据仍紧密排列；搬入 UB 后按 64 元素补齐行宽，用掩码把 y 的无效尾部清零，按真实 D 计算均值，写回仅包含有效元素。
- 多行非对齐搬运用 DataCopyPad 的 blockCount 与步长一次完成。保留原来的核间 32 字节边界分组。
- 每核只有一行、D<=4096 时使用独立小输入 kernel，四个输入合并到一次队列交接，减少小输入的固定开销。
- 6 项 CPU 检查通过，包含向量地址、非对齐搬运与写回范围、数值对比、所有 D 的 UB 和指令参数上限。最大显式 UB 用量仍为 184608 字节。
- 用户反馈 v3 全部 Pass，52.37 分。第 4/6/7 点分别为 10.26/15.16/18.43 us；第 14 点 4.05 ms。

## 已通过优化版本（v2，42.41 分）

- `D % 64 == 0` 且 `D <= 1024`：多个短行组成一批，输入/输出使用双缓冲；gamma、bias 每核加载并转换一次。用向量步长、WholeReduceSum、Brcb 和 Div 完成各行独立归一化，不从 UB 逐行 GetValue。
- 更大的 64 元素对齐 D：将整行 FP32 残差和保留在 UB，包括 D=32768。输入 x/residual 仅读取一次；FP32 的 D<=6144、FP16/BF16 的 D<=8192 时，gamma/bias 也在核内跨行复用。
- 其他 D 使用原已通过实现，继续处理非对齐的复制和核间边界。
- 内部计算仍为 FP32，输出时才转回原类型。Sqrt 后用向量 Div，未改成低精度 Rsqrt。
- 用户反馈 v2 全部 Pass，42.41 分；第 14 点从 118 ms 降至 4.05 ms，第 8 点从 379.89 us 降至 37.55 us。

## 已通过基线与回退

用户提供的评测结果：v1 为 15 点 Pass、27.45 分；v2 为 15 点 Pass、42.41 分；v3 为 15 点 Pass、52.37 分；v4 为 15 点 Pass、54.39 分；v5 为 15 点 Pass、54.80 分。源码按分数保存在 `versions/kernel_pass_*.asc`，耗时保存在相应 `versions/baseline_*.csv`。

v9 尚未提交。提交时只替换 `project/kernel.asc`；若需要回退，将 `versions/kernel_pass_54_80.asc` 的全文粘贴到平台编辑器。不要依据本地 CPU 测试宣称 NPU 已通过或已经提速。

## 原基线算法

- 一行由一个核处理。D 不超过 4096 时，残差加结果保留在 UB，归约后直接归一化；D 更大时按 4096 元素分块，两遍读取，避免大维度超过 UB。
- 下载包的 `scripts/AddRmsNormBias.py` 将 x、residual、gamma、bias 先转 float32，三步全部在 float32 中计算，最后才转回输出类型；核函数遵照这一实际判题基准。
- 用 `DataCopyPad` 处理非 32 字节长度。相邻行如果共用同一 32 字节块，划到同一个核，避免跨核写冲突。

`py -m unittest discover -s tests -v`（需 `torch`、`numpy`、`ml_dtypes`）直接调用下载包中的 golden 脚本，验证 CPU 分块算式及行分配性质，不能替代 NPU 测试。`py cannjudge_cli.py submit --problem-id 6a9a9a99bf41025d6013eb85 --project-dir E:\华为比赛\project --dry-run` 的公开模板预检查已通过，只会提交 `kernel.asc`。
#   c a n n  
 